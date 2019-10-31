import os

import torch
import re
import pickle
from tqdm import tqdm
import numpy as np

import config as cfg


def recov_from_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir):
        data_list = os.listdir(ckpt_dir)
        extension = '.ckpt'
        checkpoints = [ele for ele in data_list if(extension in ele)]
        if len(checkpoints):
            checkpoints.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
            if torch.cuda.is_available():
                model = torch.load(os.path.join(ckpt_dir, checkpoints[-1]))
            else:
                model = torch.load(ckpt_dir + checkpoints[-1], map_location='cpu')

            saved_epoch = int(re.findall(r'\d+', checkpoints[-1])[0])
            print("Resuming from epoch " + str(saved_epoch))
            return [model, saved_epoch]
        else:
            print("No checkpoints available")
    else:
        print("Can't find checkpoints directory")


def append_to_pickle_file(path, item):
    with open(path, 'ab') as file:
        pickle.dump(item, file)

def read_from_pickle_file(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def train_valid(data_loader, model, criterion, optimizer, train_flag, accuracy_flag, cuda_flag, desc):
    loss_total = 0

    one_hot_matrix = np.eye(cfg.num_classes)
    if accuracy_flag == True:
        true_possitives = np.zeros(cfg.num_classes)
        false_possitives = np.zeros(cfg.num_classes)
        false_negatives = np.zeros(cfg.num_classes)
        eps = np.full(cfg.num_classes, 1e-10)

    for ite, datapoint in enumerate(tqdm(data_loader, desc=desc)):

        input_batch = datapoint[0].type(torch.FloatTensor)
        target_batch = datapoint[1].type(torch.FloatTensor)

        if cuda_flag and torch.cuda.is_available():
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()
        
        # Forward pass
        output_batch = model(input_batch)
        batch_loss = criterion(output_batch, target_batch)
        loss_total += batch_loss.item()
        
        # Backprop
        if train_flag == True:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        if accuracy_flag == True:
            output_batch_onehot = torch.Tensor([one_hot_matrix[i] for i in output_batch.argmax(dim=1, keepdim=False)])
            for target, output in zip(target_batch.cpu(), output_batch_onehot):
                if torch.equal(target, output):
                    true_possitives += target.numpy()
                else:
                    false_possitives += output.numpy()
                    false_negatives += target.numpy()

    if accuracy_flag == True:
        precision = true_possitives / (true_possitives + false_possitives + eps)
        recall = true_possitives / (true_possitives + false_negatives + eps)
        return loss_total, precision, recall
    else:
        return loss_total
