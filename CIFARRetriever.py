import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,mean_squared_error, confusion_matrix



def split_dataset(root, val_size, train_batch, val_batch):
    # transform = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)

    # trainset = change_label(trainset)

    val_set = datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=transform)

    random_seed = 0
    valid_size = val_size
    shuffle = True
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    print("split ", split)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    train_idx, valid_idx = indices[split:], indices[:split]
    print("len train_idx ", len(train_idx), " Len valid_idx ", len(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)



    train_loader = torch.utils.data.DataLoader(trainset, 
                                           batch_size=train_batch, 
                                           sampler=train_sampler,
                                           num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_set, 
                                           batch_size=val_batch, 
                                           sampler=valid_sampler,
                                           num_workers=4)
    ## debug only:
    # for i, (images, labels) in enumerate(train_loader):
    #   print('it works!')
    return train_loader, val_loader



# def change_label(data):
#     data.targets = torch.tensor(data.targets)
#       # can fly
#     data.targets[data.targets == 0] = -1
#     data.targets[data.targets == 2] = -1

#       # cannot fly
#     data.targets[data.targets > 2] = 0
#     data.targets[data.targets == 1] = 0

#     data.targets[data.targets == -1] = 1

#     return data