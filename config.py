
import torch
from torch import nn, optim
from torchvision import transforms


# Directories
data_dir = "data/"
training_input_dir = ""
training_target_dir = ""
validation_input_dir = ""
validation_target_dir = ""

ckpt_dir = "checkpoints/"
loss_folder = "loss/"


# Data parameters
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class One_Hot(object):
    def __init__(self, num_classes):
        self.labels = torch.arange(num_classes).reshape(num_classes, 1)                      
        self.one_hot_target = (self.labels == torch.arange(num_classes).reshape(1, num_classes)).float()

    def __call__(self, tensor):
        return(self.one_hot_target[tensor])
    
target_transform = transforms.Compose([One_Hot(10)])  

training_batch_size = 100
training_shuffle = False
validation_batch_size = 10000 
validation_shuffle = False
num_workers = 2

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
lr = 0.001
