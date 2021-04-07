import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,mean_squared_error, confusion_matrix

class CnnCifar10Oracle(nn.Module):
  """ See https://tomroth.com.au/pytorch-cnn/  for model architecture"""
  def __init__(self):
    super(CnnCifar10Oracle, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.rel1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2,2)
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
    x = self.pool1(self.rel1(self.conv1(x)))
    x = self.pool2(self.rel2(self.conv2(x)))
    x = self.flat(x)
    x = self.rel3(self.fc1(x))
    x = self.rel4(self.fc2(x))
    x = self.fc3(x)  
    x = self.fc4(x)
    return torch.sigmoid(x)

def train(model, data_loader, num_epochs, learning_rate):
  """ Train the cnn"""
  printfreq = 100
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  model.train()
  for epoch in range(num_epochs):
    running_loss = 0 
    for i, (images, labels) in enumerate(data_loader):
      images = torch.autograd.Variable(images.float())
      labels = torch.autograd.Variable(labels)
      optimizer.zero_grad()
      output = model(images.float())
      labels = np.reshape(labels, (-1,1))

      # convert original labels to animal labels
      animal_label = labels.data.tolist()
      animal_label = [[1] if i[0] in [2,3,4,5,6,7] else [0] for i in animal_label]
      if i % printfreq == printfreq-1: 
        accuracy = accuracy_score(animal_label,[1 if i >= 0.5 else 0 for i in output])
      
      animal_label = torch.Tensor(animal_label)
      loss = F.binary_cross_entropy(output,animal_label)
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()
      if i % printfreq == printfreq-1:  
          print("Accuracy :", accuracy)
          print('epoch: ', epoch, "batch: ", i+1, " Running loss: ", running_loss / printfreq)
          running_loss = 0
          accuracy = 0

def create_model(train_loader, num_epochs=10, learning_rate = 0.01):
  """ Trains a cnn for cifar10 to predict if an object is animal or not. Use pre-sigmoid layer to get a low dimensional representation of the image for the oracle
    
    Inputs:
    
    train_loader - DataLoader
    
    Num_epochs - default=10
    
    learning rate - default 0.01

    Outputs: a trained CNN
  """
  print("Cifar10 model(Animal Prediction) for Oracle")
  model = CnnCifar10Oracle()
  train(model, train_loader,num_epochs,learning_rate)
  return model

def model_image_representation(model, verbose=True):
  """ Take a trained cnn model, set the gradient of parameters to false, remove the last 2 layers that does the prediction task
  
  Input: Trained CNN model
  
  Output: Trained CNN model without linear and sigmoid layers
  """
  model_layers = list(model.children())[:]
  model_features = nn.Sequential(*list(model.children())[:-2])
  for param in model_features.parameters(): 
    param.requires_grad = False
  if verbose == True:
    print("Architecture of trained input model")
    print(model_layers)
    print("\n")
    print("Architecture of model for oracle:")
    print(model_features)
  return model_features

