import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import config as cfg
from utils import read_from_pickle_file

# Server to Dell box
matplotlib.use('TkAgg')
train_loss = []
validation_loss = []

train_loss_loc = os.path.join(cfg.loss_dir, 'train_loss')
validation_loss_loc = os.path.join(cfg.loss_dir, 'validation_loss')


for item in read_from_pickle_file(train_loss_loc):
    train_loss.append(item)

for item in read_from_pickle_file(validation_loss_loc):
    validation_loss.append(item)

train_loss = np.array(train_loss)
validation_loss = np.array(validation_loss)
plt.plot(train_loss[1:-1, 0],train_loss[1:-1, 1],label="Train Loss")
plt.plot(validation_loss[1:-1, 0],validation_loss[1:-1, 1],label="Validation Loss")
plt.ylabel("Loss")
plt.xlabel("iterations")
plt.legend(loc='upper left')
plt.show()


