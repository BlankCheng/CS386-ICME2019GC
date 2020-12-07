import torch.nn as nn

from utils import FeatureExtractor


class Model(nn.Module):
    def __init__(self, input_size, encoder_name='resnet101'):
        super(Model, self).__init__()
        self.encoder_name = encoder_name
        self.size = input_size
        self.extract_list = ['layer1']
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def build_encoder(self):
        model = FeatureExtractor(self.encoder_name, self.extract_list)
        return model

    def build_decoder(self, in_channels=256):
        Dcov = nn.Sequential()
        i = 0
        while in_channels > 128:
            Dcov.add_module("Con%d" % i,
                            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3,
                                               stride=2, padding=1,
                                               output_padding=1))
            Dcov.add_module("Nd%d" % i, nn.BatchNorm2d(in_channels // 2))
            Dcov.add_module("Re%d" % i, nn.ReLU())
            in_channels //= 2
            i += 1
        Dcov.add_module("Con%d" % i,
                        nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3,
                                           stride=2, padding=1,
                                           output_padding=1))
        Dcov.add_module("Sig", nn.Tanh())
        return Dcov

    def forward(self, input):
        x = self.encoder(input)[0]
        x = self.decoder(x)
        output = self.conv1(x)
        return output
