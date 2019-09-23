

import torch
from torch.utils.data import dataloader

from utils import ImageDataset
import config as cfg


# Initialize dataset & dataloader
training_dataset = ImageDataset(data_dir=cfg.data_dir,
                                input_dir=cfg.training_input_dir,
                                target_dir=cfg.training_target_dir,
                                transform=cfg.transform)

validation_dataset = ImageDataset(data_dir=cfg.data_dir,
                                  input_dir=cfg.validation_input_dir,
                                  target_dir=cfg.validation_target_dir,
                                  transform=cfg.transform)

training_dataloader = dataloader(dataset=training_dataset,
                                 batch_size=cfg.training_batch_size,
                                 shuffle=cfg.training_shuffle)

validation_dataloader = dataloader(dataset=validation_dataset,
                                   batch_size=cfg.validation_batch_size,
                                   shuffle=cfg.validation_shuffle)

# Initialize model
if cfg.recov_from_ckpt:
    