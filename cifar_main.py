import os
import time
import sys

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import numpy as np

from utils import recov_from_ckpt, CNN_Dynamic, read_from_pickle_file, append_to_pickle_file, train_valid
import config as cfg


# Initialize dataset & dataloader
train_set = CIFAR10(root=cfg.data_dir, 
                    train=True, 
                    transform=cfg.transform, 
                    target_transform=cfg.target_transform)
train_loader = DataLoader(dataset=train_set,
                          batch_size=cfg.training_batch_size,
                          shuffle=cfg.training_shuffle, 
                          num_workers=cfg.num_workers)

validation_set = CIFAR10(root=cfg.data_dir, 
                        train=False,
                        transform=cfg.transform,
                        target_transform=cfg.target_transform)
validation_loader = DataLoader(dataset=validation_set, 
                               batch_size=cfg.validation_batch_size,
                               shuffle=cfg.validation_shuffle, 
                               num_workers=cfg.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Initialize model
if cfg.recov_from_ckpt:
    [model, saved_epoch] = recov_from_ckpt(cfg.ckpt_dir)
else:
    # Clear saved data
    user_choice = input('Are you sure you want to delete all previous trained models and data?({}/n)'.
        format("\033[4m\033[7m\033[1mY\033[0m")) # Underline Reverse Bold
    if user_choice == 'y' or user_choice == '':
        saved_models = [os.path.join(cfg.ckpt_dir, saved_model) for saved_model in os.listdir(cfg.ckpt_dir)]
        for saved_model in saved_models:
            os.remove(saved_model)

        loss_files = [os.path.join(cfg.loss_dir, loss_file) for loss_file in os.listdir(cfg.loss_dir)]
        for loss_file in loss_files:
            os.remove(loss_file)

        accuracy_files = [os.path.join(cfg.accuracy_dir, accuracy_file) for accuracy_file in os.listdir(cfg.accuracy_dir)]
        for accuracy_file in accuracy_files:
            os.remove(accuracy_file)
    elif user_choice == 'n':
        print('***Please change to recover mode in config file***')
        sys.exit()
    else:
        print('Wrong input! Exiting')
        sys.exit()

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

# Other parameters
criterion = cfg.criterion
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

# Start training
print("\nTraining Started!\n")
start_time = time.time()

for epoch in range(saved_epoch + 1, saved_epoch + 1 + cfg.training_epoch):
    print("\nEPOCH " + str(epoch) + " of " + str(saved_epoch + cfg.training_epoch) + "\n")

    # Training session
    if epoch % cfg.save_frequency == 0:
        train_loss_total, train_precision, train_recall = \
            train_valid(train_loader, model, criterion, optimizer, True, True, cfg.use_cuda, 'Train')
    else:
        train_loss_total = \
            train_valid(train_loader, model, criterion, optimizer, True, False, cfg.use_cuda, 'Train')

    train_loss_avg = train_loss_total / train_loader.__len__()

     # Validation session
    if epoch % cfg.save_frequency == 0:
        validation_loss_total, validation_precision, validation_recall = \
            train_valid(validation_loader, model, criterion, optimizer, False, True, cfg.use_cuda, 'Validation')
    else:
        validation_loss_total= \
            train_valid(validation_loader, model, criterion, optimizer, False, False, cfg.use_cuda, 'Validation')

    validation_loss_avg = validation_loss_total / validation_loader.__len__()
    
    # Printing session
    print('Epoch:', epoch, 
          '\tTime since start:', int((time.time()-start_time))//60, 'min', int((time.time()-start_time)%60), 'sec')
    print('Train Loss Avg:', train_loss_avg)
    print('Validation Loss Avg:', validation_loss_avg)
    if epoch % cfg.save_frequency == 0:
        print('Validation Precision', validation_precision)
        print('Validation Recall', validation_recall)

    # Saving Session
    # Save average loss value to file
    train_loss_loc = os.path.join(cfg.loss_dir, 'train_loss')
    append_to_pickle_file(train_loss_loc, [epoch, train_loss_avg])
    validation_loss_loc = os.path.join(cfg.loss_dir, 'validation_loss')
    append_to_pickle_file(validation_loss_loc, [epoch, validation_loss_avg])
    
    if epoch % cfg.save_frequency == 0:
        # Save precision & recall
        validation_precision_loc = os.path.join(cfg.accuracy_dir, 'validation_precision')
        append_to_pickle_file(validation_precision_loc, [epoch, validation_precision])
        validation_recall_loc = os.path.join(cfg.accuracy_dir, 'validation_recall')
        append_to_pickle_file(validation_recall_loc, [epoch, validation_recall])
        # Save model
        ckpt_dir = os.path.join(cfg.ckpt_dir, 'model_epoch_' + str(epoch) + '.ckpt')
        torch.save(model, ckpt_dir)
        print("\nmodel saved at epoch : " + str(epoch) + "\n")

        
#    scheduler.step() #Decrease learning rate
