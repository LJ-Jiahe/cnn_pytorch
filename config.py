
from torch import nn


# Directories
data_dir = "data/"
training_input_dir = ""
training_target_dir = ""
validation_input_dir = ""
validation_target_dir = ""

ckpt_dir = "checkpoints/"
loss_folder = "loss/"


# Data parameters
transform=None

training_batch_size = 50
training_shuffle = False
validation_batch_size = 1
validation_shuffle = False

# Model parameters
recov_from_ckpt = False

channel_num = [3, 16, 64, 128]
kernel_sizes = [3, 3, 3]
strides = [1, 1, 1]
paddings = [1, 1, 1]
ac_funs = [nn.ReLU, nn.ReLU, nn.ReLU]
img_size = 32
fc_sizes = [channel_num[-1] * img_size**2, 1024, 10]
fc_ac_funs = [nn.ReLU, None]




training_epoch = 1000

criterion = nn.MSELoss()
lr = 0.01