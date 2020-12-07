from torch import nn, flatten
from torchvision import models


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


if __name__ == '__main__':
    from torch import rand

    extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
    extract_result = FeatureExtractor('resnet101', extract_list)

    x_tensor = rand((10, 3, 224, 224))
    extract_result(x_tensor)


def calculate_sauc(smaps, smaps_pred):
    """
    Calculate shuffled AUC.
    """
    pass


def calculate_auc_b(smaps, smaps_pred):
    """
    Calculate AUC-Borji.
    """
    pass


def calculate_auc_j(smaps, smaps_pred):
    """
    Calculate AUC-Judd.
    """
    pass


def calculate_nss(smaps, smaps_pred):
    """
    Calculate the normalized scanpath saliency.
    """
    pass


def calculate_cc(smaps, smaps_pred):
    """
    Calculate correlation coefficient.
    """
    pass


def calculate_sim(smaps, smaps_pred):
    """
    Calculate similarities.
    """
    pass


def calculate_kld(smaps, smaps_pred):
    """
    Calculate KL-Divergence.
    """
    pass
