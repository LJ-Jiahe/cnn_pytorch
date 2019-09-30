

from torchvision.datasets import CIFAR10
from torch.utils.data import Dataloader

import config as cfg


test_set = CIFAR10(root=cfg.data_dir,
                   train=False,
                   transform=cfg.transform,
                   target_transform=cfg.target_transform)
test_loader = Dataloader(dataset=test_set,
                         batch_size=cfg.test_batch_size,
                         shuffle=cfg.test_shuffle,
                         num_workders=cfg.num_workders)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Recover model from checkpoints
[model, saved_epoch] = recov_from_ckpt(cfg.chpt_dir)

if torch.cuda.is_available():
    print("\nCuda available\n")
    model.cuda()

test_loss
