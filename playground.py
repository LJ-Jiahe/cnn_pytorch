

import config as cfg
from utils import ImageDataset, CNN_Sequential, CNN_Dynamic
from torch import nn

channel_num = [3, 16, 64, 128]
kernel_sizes = [3, 3, 3]
strides = [1, 1, 1]
paddings = [1, 1, 1]
ac_funs = [nn.ReLU, nn.ReLU, nn.ReLU]
img_size = 28
n_classes = 10

model = CNN_Dynamic(channel_num, kernel_sizes, strides, paddings, ac_funs, img_size, n_classes)
print(model)