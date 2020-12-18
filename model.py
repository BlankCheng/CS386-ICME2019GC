import torch.nn as nn
from torch import cat
from utils import FeatureExtractor


class Model(nn.Module):
    def __init__(self, input_size, encoder_name='resnet101'):
        super(Model, self).__init__()
        self.encoder_name = encoder_name
        self.size = input_size
        self.extract_list = ['layer2', 'layer3', 'layer4']
        self.encoder = self.build_encoder()
        self.decoder1 = self.build_decoder(in_channels=512)
        self.decoder2 = self.build_decoder(in_channels=1024)
        self.decoder3 = self.build_decoder(in_channels=2048)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.Sigmoid()

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
        x1, x2, x3 = self.encoder(input)
        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        x3 = self.decoder3(x3)
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x3 = self.conv1(x3)
        x = cat([x1, x2, x3], dim=1)
        print(x.shape)
        output = self.conv2(x)
        output = self.relu(output)
        return output