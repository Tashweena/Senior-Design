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


class PredictorCNNCIFAR(nn.Module):
    def __init__(self):
        super(PredictorCNNCIFAR, self).__init__()
        ### https://tomroth.com.au/pytorch-cnn/ 
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # # we use the maxpool multiple times, but define it once
        # self.pool = nn.MaxPool2d(2,2)
        # # in_channels = 6 because self.conv1 output 6 channel
        # self.conv2 = nn.Conv2d(6,16,5) 
        # # 5*5 comes from the dimension of the last convnet layer
        # self.fc1 = nn.Linear(16*5*5, 120) 
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        # self.fc4 = nn.Linear(10, 1) 

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.rel1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        # in_channels = 6 because self.conv1 output 6 channel
        self.conv2 = nn.Conv2d(6,16,5) 
        self.rel2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        self.flat = nn.Flatten(1)
        self.fc1 = nn.Linear(16*5*5, 120) 
        self.rel3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.rel4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 1) 

    def forward(self, x):
        # print("here ???")
        x = self.pool1(self.rel1(self.conv1(x)))
        x = self.pool2(self.rel2(self.conv2(x)))
        x = self.flat(x)
        x = self.rel3(self.fc1(x))
        x = self.rel4(self.fc2(x))
        x = self.fc3(x)  
        x = self.fc4(x)
        return torch.sigmoid(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16*5*5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)  
        # x = self.fc4(x)
        # return torch.sigmoid(x)

    def fit(self, x_train, y_train):
        return

    ### don't need this function
    def predict(self, data):
        y_pred = self.forward(data)
        y_pred = y_pred.detach().numpy()
        y_pred = y_pred.flatten()
        return y_pred

#### train the model 
def train(model, data_loader, num_epochs, learning_rate):
  running_loss = 0 
  printfreq = 100
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  # criterion = nn.CrossEntropyLoss()
  model.train()
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
      images = torch.autograd.Variable(images.float())
      labels = torch.autograd.Variable(labels)
      optimizer.zero_grad()
      output = model(images.float())
      labels = np.reshape(labels, (-1,1))

      animal_label = labels.data.tolist()
      animal_label = [[1] if i[0] in [2,3,4,5,6,7] else [0] for i in animal_label]
      if i % printfreq == printfreq-1: 
        accuracy = accuracy_score(animal_label,[1 if i >= 0.5 else 0 for i in output])
      
      animal_label = torch.Tensor(animal_label)
      loss = F.binary_cross_entropy(output,animal_label)
      loss.backward()
      optimizer.step()
      
      # if (i % 100) == 0:
      #   print('num_correct ', sum((output < 0.5)== (animal_label < 0.5)))
      running_loss += loss.item()
      if i % printfreq == printfreq-1:  
          print("Accuracy :", accuracy)
          print('epoch: ', epoch, "batch: ", i+1, " Running loss: ", running_loss / printfreq)
          running_loss = 0
          accuracy = 0

def create_model(train_loader, num_epochs=2, learning_rate = 0.01):
  model = PredictorCNNCIFAR()
  print("Another NN trained on CIFAR")
  train(model, train_loader,num_epochs,learning_rate)
  return model
