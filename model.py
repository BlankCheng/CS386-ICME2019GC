import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import FeatureExtractor

class Model(nn.Module):
    def __init__(self, input_size, encoder_name='resnet101'):
        super(Model, self).__init__()
        self.encoder_name = encoder_name
        self.size = input_size
        if encoder_name == "resnet101":
            self.extract_list = ['layer4']
            self.channels = [2048]
        elif encoder_name == "vgg16":
            self.extract_list = ["15", "22", "relu3"]
            self.channels = [256, 512, 512]
        else:
            raise Exception("unknown encoder name")
        self.encoder = self.build_encoder()
        self.decoder_list = []
        for i_ in range(len(self.extract_list)):
            self.decoder_list.append(self.build_decoder(in_channels=self.channels[i_]))
        self.decoder_list = nn.ModuleList(self.decoder_list)

        self.conv1 = nn.Conv2d(in_channels=64 * len(self.extract_list), out_channels=1, kernel_size=3, stride=1, padding=1)

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
        Xs = self.encoder(input)
        x_out = []
        for i in range(len(self.extract_list)):
            x_out.append(self.decoder_list[i](Xs[i]))
        x_cat = torch.cat(x_out, dim=1)
        # print("concat size", x_cat.size())
        output = torch.sigmoid(self.conv1(x_cat))
        return output


# class Model(nn.Module):
#     def __init__(self, input_size, encoder_name='resnet101'):
#         super(Model, self).__init__()
#         self.encoder_name = encoder_name
#         self.size = input_size
#         self.extract_list = ['layer4']
#         self.encoder = self.build_encoder()
#         self.decoder = self.build_decoder(in_channels=2048)
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
#         self.relu = nn.Sigmoid()
#
#     def build_encoder(self):
#         model = FeatureExtractor(self.encoder_name, self.extract_list)
#         return model
#
#     def build_decoder(self, in_channels=256):
#         Dcov = nn.Sequential()
#         i = 0
#         while in_channels > 64:
#             Dcov.add_module("Con%d" % i,
#                             nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3,
#                                                stride=2, padding=1,
#                                                output_padding=1))
#             Dcov.add_module("Nd%d" % i, nn.BatchNorm2d(in_channels // 2))
#             Dcov.add_module("Re%d" % i, nn.ReLU())
#             in_channels //= 2
#             i += 1
#         return Dcov
#
#     def forward(self, input):
#         x = self.encoder(input)[0]
#         x = self.decoder(x)
#         output = F.sigmoid(self.conv1(x))
#         return output
