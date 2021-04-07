import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def split_dataset(root, val_size, train_batch,val_batch,transform=None, verbose=True):
    """ Split the CIFAR10 training dataset into validation and training  data.
    
    Input:
    root - str, folder to download data to
    
    val_size - size of validation data
    
    train_batch - int, batch size for train data
    
    val_batch - int, batch size for validation data
    
    transform(optional) - torchvision.transforms to apply to the data

    Output: Train Dataloader, Validation Dataloader
    """
    if transform == None:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    # download the dataset
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10(root=root, train=True,download=True, transform=transform)

    random_seed = 0
    valid_size = val_size
    shuffle = True
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # split the train data into train and val
    train_idx, valid_idx = indices[split:], indices[:split]
    if verbose == True:
        print("Size of training data ", len(train_idx), " Size of validation data ", len(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(trainset, 
                                           batch_size=train_batch, 
                                           sampler=train_sampler,
                                           num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_set, 
                                           batch_size=val_batch, 
                                           sampler=valid_sampler,
                                           num_workers=1)
    return train_loader, val_loader