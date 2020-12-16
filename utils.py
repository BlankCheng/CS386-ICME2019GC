from torch import nn, flatten
from torchvision import models
import numpy as np
import torch
import random
import math


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name
        self.submodule = self.load_model()
        self.extracted_layers = extracted_layers

    def load_model(self):
        model = None
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif self.model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
        else:
            print("Model name is unknown")
        return model

    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        t = 0
        for name, module in self.submodule._modules.items():
            if name == "classifier" or name == "fc": x = flatten(x, 1)
            x = module(x)
            # print(name, x.shape)
            if name in self.extracted_layers:
                outputs.append(x)
                t += 1
                if t == len(self.extracted_layers):
                    return outputs
        return outputs


def generate_dummy(size=14,num_fixations=100,num_salience_points=200):
    # first generate dummy gt and salience map
    discrete_gt = np.zeros((size,size))
    s_map = np.zeros((size,size))

    for i in range(0,num_fixations):
        discrete_gt[np.random.randint(size),np.random.randint(size)] = 1.0

    for i in range(0,num_salience_points):
        s_map[np.random.randint(size),np.random.randint(size)] = 255*round(random.random(),1)
    # check if gt and s_map are same size
    assert discrete_gt.shape==s_map.shape, 'sizes of ground truth and salience map don\'t match'
    return s_map,discrete_gt


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - torch.min(s_map))/((torch.max(s_map) - torch.min(s_map)) * 1.0)
    return norm_s_map


def discretize_gt(gt):
    import warnings
    warnings.warn('can improve the way GT is discretized')
    return gt / 255


def calculate_sauc(gt_, s_map_, other_map_=None, splits=100):
    """
    Calculate shuffled AUC.
    """
    batch_size = gt_.size(0)
    gt_ = torch.where(gt_ > 0, torch.ones_like(gt_), torch.zeros_like(gt_))
    other_map_ = gt_

    ret_ = 0

    for i_ in range(batch_size):

        gt = gt_[i_].detach().cpu().squeeze(0).numpy()
        s_map = s_map_[i_].detach().cpu().squeeze(0).numpy()
        other_map = other_map_[i_].detach().cpu().squeeze(0).numpy()

        num_fixations = np.sum(gt)

        x, y = np.where(other_map == 1)
        other_map_fixs = []
        for j in zip(x, y):
            other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
        ind = len(other_map_fixs)
        assert ind == np.sum(other_map), 'something is wrong in auc shuffle'

        num_fixations_other = min(ind, num_fixations)

        num_pixels = s_map.shape[0] * s_map.shape[1]
        random_numbers = []
        for i in range(0, splits):
            temp_list = []
            t1 = np.random.permutation(ind)
            for k in t1:
                temp_list.append(other_map_fixs[k])
            random_numbers.append(temp_list)

        aucs = []
        # for each split, calculate auc
        for i in random_numbers:
            r_sal_map = []
            for k in i:
                # print(k, s_map.shape[0])
                r_sal_map.append(s_map[k % s_map.shape[0] - 1, k // s_map.shape[0]])
            # in these values, we need to find thresholds and calculate auc
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            r_sal_map = np.array(r_sal_map)

            # once threshs are got
            thresholds = sorted(set(thresholds))
            area = []
            area.append((0.0, 0.0))
            for thresh in thresholds:
                # in the salience map, keep only those pixels with values above threshold
                temp = np.zeros(s_map.shape)
                temp[s_map >= thresh] = 1.0
                num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
                tp = num_overlap / (num_fixations * 1.0)

                # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
                # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
                fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

                area.append((round(tp, 4), round(fp, 4)))

            area.append((1.0, 1.0))
            area.sort(key=lambda x: x[0])
            tp_list = [x[0] for x in area]
            fp_list = [x[1] for x in area]

            aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

        ret_ += np.mean(aucs)
    return ret_ / batch_size


def calculate_auc_b(gt_, s_map_, other_map_=None, splits=100):
    """
    Calculate AUC-Borji.
    """
    pass


def calculate_auc_j(gt_, s_map_):
    """
    Calculate AUC-Judd.
    """
    batch_size = gt_.size(0)

    ret_ = 0.

    gt_ = torch.where(gt_ > 0, torch.ones_like(gt_), torch.zeros_like(gt_))

    for i_ in range(batch_size):

        #  ===========================
        # print(torch.max(s_map_), torch.max(gt_))

        gt = gt_[i_].detach().cpu().squeeze(0).numpy()
        s_map = s_map_[i_].detach().cpu().squeeze(0).numpy()
        # ground truth is discrete, s_map is continous and normalized
        # thresholds are calculated from the salience map, only at places where fixations are present
        thresholds = []
        for i in range(0, gt.shape[0]):
            for k in range(0, gt.shape[1]):
                if gt[i][k] > 0:
                    thresholds.append(s_map[i][k])

        num_fixations = np.sum(gt)
        # num fixations is no. of salience map values at gt >0

        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            assert np.max(gt) <= 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
            assert np.max(s_map) <= 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
            # this becomes nan when gt is full of fixations..this won't happen
            fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        # print(np.trapz(np.array(tp_list), np.array(fp_list)))
        ret_ += np.trapz(np.array(tp_list), np.array(fp_list))
    return ret_ / batch_size


def calculate_nss(gt, s_map):
    """
    Calculate the normalized scanpath saliency.
    """
    # gt = discretize_gt(gt)
    # print(torch.max(gt), gt.size())
    s_mu = torch.mean(s_map, dim=[1, 2, 3]).view(-1, 1, 1, 1)
    s_std = torch.std(s_map, dim=[1, 2, 3]).view(-1, 1, 1, 1)
    s_map_norm = (s_map - s_mu) / s_std
    nss = torch.where(s_map_norm * gt > 0, s_map_norm, torch.zeros_like(s_map_norm))
    tot = torch.sum(nss > 0)
    # print(tot)
    return torch.sum(nss) / tot


def calculate_cc(gt, s_map):
    """
    Calculate correlation coefficient.
    """
    s_mu = torch.mean(s_map, dim=[1, 2, 3]).view(-1, 1, 1, 1)
    s_std = torch.std(s_map, dim=[1, 2, 3]).view(-1, 1, 1, 1)
    gt_mu = torch.mean(gt, dim=[1, 2, 3]).view(-1, 1, 1, 1)
    gt_std = torch.std(gt, dim=[1, 2, 3]).view(-1, 1, 1, 1)
    s_map_norm = (s_map - s_mu) / s_std
    gt_norm = (gt - gt_mu) / gt_std
    a = s_map_norm
    b = gt_norm
    r = torch.sum(a * b, dim=[1, 2, 3]) / torch.sqrt(torch.sum(a * a, dim=[1, 2, 3]) * torch.sum(b * b, dim=[1, 2, 3]))
    return torch.mean(r)


def calculate_sim(gt, s_map):
    """
    Calculate similarities.
    """
    s_map = normalize_map(s_map)
    gt = normalize_map(gt)
    s_map = s_map / torch.sum(s_map)
    gt = gt / torch.sum(gt)
    return torch.sum(torch.min(gt, s_map))


def calculate_kld(gt, s_map):
    """
    Calculate KL-Divergence.
    """
    s_map = s_map / torch.sum(s_map)
    gt = gt / torch.sum(gt)
    eps = 2.2204e-16
    return torch.sum(gt * torch.log(eps + gt / (s_map + eps)))


if __name__ == '__main__':
    from torch import rand

    extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
    extract_result = FeatureExtractor('resnet101', extract_list)

    x_tensor = rand((10, 3, 224, 224))
    extract_result(x_tensor)
