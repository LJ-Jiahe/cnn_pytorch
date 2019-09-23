

from torch import nn
from torch.nn import ModuleList



class CNN_Module(nn.Module):
    def __init__(self, hyper_params):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=, 
                               out_channels=, 
                               kernel_size=, 
                               stride=1, 
                               padding=0)
        self.conv2 = nn.Conv2d()
    def forward(self):


class CNN_Dynamic(nn.module):
    