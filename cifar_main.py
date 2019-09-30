import os
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from utils import recov_from_ckpt, CNN_Dynamic, read_from_pickle_file, append_to_pickle_file
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

    validation_loss_total = 0
    for validation_ite, validation_datapoint in enumerate(tqdm(validation_loader, desc='Validation')):
        validation_input_batch = validation_datapoint[0].type(torch.FloatTensor)
        validation_target_batch = validation_datapoint[1].type(torch.FloatTensor)
    
        if torch.cuda.is_available():
            validation_input_batch = validation_input_batch.cuda()
            validation_target_batch = validation_target_batch.cuda()
        
        validation_output_batch = model(validation_input_batch)
        validation_loss = criterion(validation_output_batch, validation_target_batch)
        validation_loss_total += validation_loss.item()

    validation_loss_avg = validation_loss_total / validation_loader.__len__()
    validation_loss_loc = os.path.join(cfg.loss_folder, 'validation_loss')
    append_to_pickle_file(validation_loss_loc, [epoch, validation_loss_avg])
    

    

    train_loss_total = 0
    for train_ite, train_datapoint in enumerate(tqdm(train_loader, desc='Train')):
        train_input_batch = train_datapoint[0].type(torch.FloatTensor)
        train_target_batch = train_datapoint[1].type(torch.FloatTensor)

        if torch.cuda.is_available():
            train_input_batch = train_input_batch.cuda()
            train_target_batch = train_target_batch.cuda()

        optimizer.zero_grad()
        train_output_batch = model(train_input_batch)
        train_loss = criterion(train_output_batch, train_target_batch)
        train_loss_total += train_loss.item()
        train_loss.backward()
        optimizer.step()
    
    # # Write average loss value to file once every epoch
    train_loss_loc = os.path.join(cfg.loss_folder, 'train_loss')
    train_loss_avg = train_loss_total / train_loader.__len__()
    append_to_pickle_file(train_loss_loc, [epoch, train_loss_avg])

# Print Loss
    time_since_start = (time.time()-start_time) / 60
    print('\nEpoch: {} \nLoss avg: {} \nValidation Loss avg: {} \nTime(mins) {}'.format(
         epoch, train_loss_avg, validation_loss_avg, time_since_start))

# Save every 10 epochs
    if epoch % 100 == 0:
        ckpt_folder = os.path.join(cfg.ckpt_dir, 'model_epoch_' + str(epoch) + '.pt')
        torch.save(model, ckpt_folder)
        print("\nmodel saved at epoch : " + str(epoch) + "\n")
    
#    scheduler.step() #Decrease learning rate
