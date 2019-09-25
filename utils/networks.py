

from torch import nn
from torch.nn import ModuleList


# Example of CNN model with fixed structure
class CNN_2_2_fixed(nn.Module):
    def __init__(self, in_channels, img_size, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=32, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, 
                               out_channels=128, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * img_size * img_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.ac_fun1 = nn.ReLU()
        self.ac_fun2 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.ac_fun1(self.bn1(self.conv1(x)))
        x = self.ac_fun1(self.bn2(self.conv2(x)))
        
        x = x.view(x.size(0), -1) # Leaving size[0] for batch

        x = self.ac_fun2(self.fc1(x))
        x = self.fc2(x)

        return x


# Example of CNN with Sequential
class CNN_2_2_Sequential(nn.Module):
    def __init__(self, in_channels, img_size,  n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            conv_layer(in_channels=in_channels, 
                       out_channels=32, 
                       kernel_size=3, 
                       stride=1, 
                       padding=1,
                       ac_fun=nn.ReLU),
            conv_layer(in_channels=32, 
                       out_channels=128, 
                       kernel_size=3, 
                       stride=1, 
                       padding=1,
                       ac_fun=nn.ReLU)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * img_size * img_size, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )
        

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Dynamically generated CNN model
class CNN_Dynamic(nn.Module):
    def __init__(self, channel_num, kernel_sizes, strides, paddings, ac_funs, 
                 img_size, fc_sizes, fc_ac_funs):
        super().__init__()

        conv_params = zip(channel_num, channel_num[1:], kernel_sizes, strides, 
                         paddings, ac_funs)
        conv_layers = \
            [conv_layer(in_channels, out_channels, kernel_size, stride, padding, ac_fun)
            for in_channels, out_channels, kernel_size, stride, padding, ac_fun 
            in conv_params]
        # print(conv_layers)
        self.conv = nn.Sequential(*conv_layers)
        
        fc_params = zip(fc_sizes, fc_sizes[1:], fc_ac_funs)
        fc_layers = \
            [fc_layer(in_size, out_size, ac_fun)
            for in_size, out_size, ac_fun
            in fc_params]
        # print(fc_layers)
        self.fc = nn.Sequential(*fc_layers)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Create a layer of convolution
def conv_layer(in_channels, out_channels, kernel_size, stride, padding, ac_fun):
    conv_layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, 
                  out_channels=out_channels, 
                  kernel_size=kernel_size, 
                  stride=stride, 
                  padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        ac_fun()
    )

    return conv_layer


# Create a layer of mlp
def fc_layer(in_size, out_size, ac_fun):

    fc_layer = nn.Sequential(
        nn.Linear(in_size, out_size)
    )
    if ac_fun != None:
        fc_layer.add_module(str(fc_layer.__len__()), ac_fun())

    return fc_layer