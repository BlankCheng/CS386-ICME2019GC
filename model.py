import torch.nn as nn
from torch import cat

from back_model.nonlocalblock import NonLocalBlock
from utils import FeatureExtractor


class Model(nn.Module):
    def __init__(self, input_size, encoder_name='resnet101', extract_list=None, channels=None):
        super(Model, self).__init__()
        if channels is None:
            channels = [512, 1024, 2048]
        if extract_list is None:
            extract_list = ['layer2', 'layer3', 'layer4']
        self.channels = channels
        self.encoder_name = encoder_name
        self.size = input_size
        self.extract_list = extract_list
        self.encoder = self.build_encoder()
        self.localblock = []
        for i_ in self.channels:
            self.localblock.append(NonLocalBlock(i_, i_, i_ // 4))
        self.localblock = nn.ModuleList(self.localblock)
        self.decoder = []
        for i_ in self.channels:
            self.decoder.append(self.build_decoder(in_channels=i_))
        self.decoder_list = nn.ModuleList(self.decoder)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=len(self.channels), out_channels=1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()

    def build_encoder(self):
        model = FeatureExtractor(self.encoder_name, self.extract_list)
        return model

    def build_decoder(self, in_channels=256):
        Dcov = nn.Sequential()
        i = 0
        while in_channels > 64:
            Dcov.add_module("Con%d" % i,
                            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3,
                                               stride=2, padding=1,
                                               output_padding=1))
            Dcov.add_module("Nd%d" % i, nn.BatchNorm2d(in_channels // 2))
            Dcov.add_module("Re%d" % i, nn.ReLU())
            in_channels //= 2
            i += 1
        return Dcov

    def forward(self, input):
        x_e = self.encoder(input)
        x_block = []
        for _ in range(len(x_e)):
            x_block.append(self.localblock[_](x_e[_]))
        x_out = []
        for _ in range(len(x_block)):
            x_out.append(self.conv1(self.decoder_list[_](x_block[_])))
        x = cat(x_out, dim=1)
        # print(x.shape)
        output = self.conv2(x)
        output = self.activation(output)
        return output


if __name__ == '__main__':
    import torch
    from torch import rand

    a = True
    device = torch.device("cuda" if a else "cpu")
    x_tensor = rand((10, 3, 224, 224)).to(device)
    net = Model(input_size=(3, 224, 224), encoder_name='densenet169', extract_list=['denseblock1', 'denseblock2'],
                channels=[256, 512]).to(device)
    print(x_tensor)
    y = net(x_tensor)
    print(y)
