

from torch.utils.data import DataLoader
from torch import nn
import torch

from utils import ImageDataset, recov_from_ckpt, CNN_Dynamic
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

training_dataloader = DataLoader(dataset=training_dataset,
                                 batch_size=cfg.training_batch_size,
                                 shuffle=cfg.training_shuffle)

validation_dataloader = DataLoader(dataset=validation_dataset,
                                   batch_size=cfg.validation_batch_size,
                                   shuffle=cfg.validation_shuffle)

# Initialize model
if cfg.recov_from_ckpt:
    [model, saved_epoch] = recov_from_ckpt(cfg.ckpt_dir)
else:
    model = CNN_Dynamic(channel_num=cfg.channel_num, 
                        kernel_sizes=cfg.kernel_sizes, 
                        strides=cfg.strides, 
                        paddings=cfg.paddings, 
                        ac_funs=cfg.ac_funs, 
                        img_size=cfg.img_size, 
                        fc_sizes=cfg.fc_sizes, 
                        fc_ac_funs=cfg.fc_ac_funs)
    saved_epoch = 0

if torch.cuda.is_available():
    print("\nCuda available\n")
    model.cuda()


# Start training
# for epoch in range(saved_epoch + 1, saved_epoch + 1 + cfg.training_epoch):
    