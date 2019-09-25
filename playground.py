

import config as cfg
from utils import ImageDataset, CNN_Dynamic
from torch import nn

channel_num = [3, 16, 64, 128]
kernel_sizes = [3, 3, 3]
strides = [1, 1, 1]
paddings = [1, 1, 1]
ac_funs = [nn.ReLU, nn.ReLU, nn.ReLU]
img_size = 28
fc_sizes = [channel_num[-1] * img_size * img_size, 1024, 10]
fc_ac_funs = [nn.Sigmoid, None]

model = CNN_Dynamic(
    channel_num, kernel_sizes, strides, paddings, ac_funs, img_size, fc_sizes, fc_ac_funs)
print(model)