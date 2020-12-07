from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch

dconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1, output_padding=0,
                           bias=False)
init.constant(dconv.weight, 1)
print(dconv.weight)

input = Variable(torch.ones(1, 1, 2, 2))
print(input)
print(dconv(input))
