import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import config as cfg
from utils import read_from_pickle_file


matplotlib.use('TKAgg')
train_loss = []
test_loss = []

train_loss_loc = os.path.join(cfg.loss_folder, 'train_loss')
test_loss_loc = os.path.join(cfg.loss_folder, 'validation_loss')


for item in read_from_pickle_file(train_loss_loc):
    train_loss.append(item)

for item in read_from_pickle_file(test_loss_loc):
    test_loss.append(item)

train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
plt.plot(train_loss[0:-1, 0],train_loss[0:-1, 1],label="Training Loss")
plt.plot(test_loss[0:-1, 0],test_loss[0:-1, 1],label="Testing Loss")
plt.ylabel("Loss")
plt.xlabel("iterations")
plt.legend(loc='upper left')
plt.show()
