

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch import optim

import config as cfg
from utils import recov_from_ckpt, train_valid


test_set = CIFAR10(root=cfg.data_dir,
                   train=False,
                   transform=cfg.transform,
                   target_transform=cfg.target_transform)
test_loader = DataLoader(dataset=test_set,
                         batch_size=cfg.test_batch_size,
                         shuffle=cfg.test_shuffle,
                         num_workers=cfg.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
one_hot = np.eye(10)

# Recover model from checkpoints
[model, saved_epoch] = recov_from_ckpt(cfg.ckpt_dir)

if cfg.use_cuda and torch.cuda.is_available():
    print("\nCuda available\n")
    model.cuda()
else:
    model.cpu()

# Other parameters
criterion = cfg.criterion
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

loss_total, precision, recall = \
    train_valid(test_loader, model, criterion, optimizer, True, True, cfg.use_cuda, 'Test')

print(precision)
print(recall)