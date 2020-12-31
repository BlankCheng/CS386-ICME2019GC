import argparse
import os

from PIL import Image
from torchvision import transforms

from dataloader import preprocess
from model import Model
from utils import *


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/NAS2020/Share/chenxianyu/PycharmProjects/CS386-ICME2019GC')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./checkpoints')
    parser.add_argument('--dataset', type=str, choices=['EyeTracker', 'MIT1003'], default='EyeTracker')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


unloader = transforms.ToPILImage()


def save_image(tensor, name, dir):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not os.path.exists(dir):
        os.makedirs(dir)
    image.save(os.path.join(dir, name))


if __name__ == '__main__':
    args = arg_parse()
    device = torch.device("cuda" if args.cuda else "cpu")

    test_data_path = os.path.join(args.data_path, 'test_data/ICME_GC_test/task1/Image')
    out_path = os.path.join(args.data_path, 'our_result')
    img_files = os.listdir(test_data_path)

    net = Model(input_size=(3, 224, 224), encoder_name='se_resnext101', extract_list=['layer2', 'layer3', 'layer4'],
                channels=[512, 1024, 2048])
    best_epoch = 73
    net.load_state_dict(torch.load(os.path.join(args.model_path, 'se_resnext_best_{}.pth'.format(best_epoch))))
    net = net.to(device)
    for file in img_files:
        img = Image.open(os.path.join(test_data_path, file))
        img = preprocess(img)
        img = torch.unsqueeze(img, 0)
        img=img.to(device)
        smap_pred = net(img)
        smap_pred = (smap_pred - smap_pred.min()) / (smap_pred.max() - smap_pred.min())
        save_image(smap_pred, '{}.png'.format(file.split('.')[0] + '_s'), out_path)
