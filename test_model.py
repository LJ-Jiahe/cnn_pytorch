

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import config as cfg
from utils import recov_from_ckpt


test_set = CIFAR10(root=cfg.data_dir,
                   train=False,
                   transform=cfg.transform,
                   target_transform=cfg.target_transform)
test_loader = DataLoader(dataset=test_set,
                         batch_size=cfg.test_batch_size,
                         shuffle=cfg.test_shuffle,
                         num_workers=cfg.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Recover model from checkpoints
[model, saved_epoch] = recov_from_ckpt(cfg.ckpt_dir)

if torch.cuda.is_available():
    print("\nCuda available\n")
    model.cuda()

test_loss_total = 0
true_positive = np.zeros(cfg.num_classes)
false_positive = np.zeros(cfg.num_classes)
false_negative = np.zeros(cfg.num_classes)

for test_ite, test_datapoint in enumerate(tqdm(test_loader, desc='Validation')):
    test_input_batch = test_datapoint[0]#.type(torch.FloatTensor)
    test_target_batch = test_datapoint[1]#.type(torch.FloatTensor)

    if torch.cuda.is_available():
        test_input_batch = test_input_batch.cuda()
        test_target_batch = test_target_batch.cuda()

    test_output_batch = model(test_input_batch)
    print(test_output_batch.shape)

    
