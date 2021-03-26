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


class PredictorCNN(nn.Module):
    def __init__(self):
        super(PredictorCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining first 2D convolution layer
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.fc1 = nn.Linear(8*8*16,120)
        # self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.fc4 = nn.Linear(10,1)
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(self.fc4(x))
        # x = torch.sigmoid(self.fc3(x))
        # print(" -------", x)
        return x

    def fit(self, x_train, y_train):
        return
        # dummy function 

    def predict(self, data):
        y_pred = self.forward(data)
        y_pred = y_pred.detach().numpy()
        y_pred = y_pred.flatten()
        return y_pred
  
### Train the network 
def train(model, data_loader, num_epochs, learning_rate):
    ## Loss function and optimizer 
    # criterion =  F.binary_cross_entropy()
    # nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    model.train()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = torch.autograd.Variable(images.float())
            labels = torch.autograd.Variable(labels)
            labels = np.reshape(labels, (-1,1))
            # labels = labels.to(torch.float32)
            optimizer.zero_grad()
            outputs = model(images.float())

            fly_label = labels.data.tolist()
            fly_label = [[1] if i[0] in [0,2] else [0] for i in fly_label ]
            fly_label = torch.Tensor(fly_label)

            #animal_label = labels.data.tolist()
            #animal_label = [[1] if i[0] in [2,3,4,5,6,7] else [0] for i in animal_label]
            #animal_label = torch.Tensor(animal_label)

            
            # print("outputs shape: ", outputs.shape)
            # print("labels shape: ", labels.shape)
            # print(labels)
            # print(outputs)
  
            loss =  F.binary_cross_entropy(outputs, fly_label)
            # criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                
                y_true, y_predicted, y_pred_labels = evaluate(model, data_loader)


                print(f'Epoch : {epoch}/{num_epochs}, '
                      f'Iteration : {i}/{len(data_loader)},  '
                      f'Loss: {loss.item():.4f},',
                      f'Mean Squared Error: {mean_squared_error(y_true, y_predicted):.4f},',
                      f'Train Accuracy: {100.* accuracy_score(y_true, y_pred_labels):.4f},')


### Evaluate the model for training
def evaluate(model, data_loader):
    model.eval()
    y_true = []
    y_predicted = []
    y_pred_labels = []
  
    for images, labels in data_loader:
        images = torch.autograd.Variable(images.float())
        labels = np.reshape(labels, (-1,1))
        #animal_label = labels.data.tolist()
        #animal_label = [[1] if i[0] in [2,3,4,5,6,7] else [0] for i in animal_label]
        fly_label = labels.data.tolist()
        fly_label = [1 if i[0] in [0,2] else 0 for i in fly_label]

        outputs = model(images) ### these are the probabilities
        # print(outputs)

        ##### convert to labels 0 and 1 
        pred = outputs.data.tolist()
        pred = [[1] if e[0] >= 0.5 else [0] for e in pred]
        pred = torch.Tensor(pred)
        #####

        # _, predicted = torch.max(outputs.data,0)

        y_true.extend(fly_label)
        y_predicted.extend(outputs)
        y_pred_labels.extend(pred)

        # print(y_true)
        # print("  PREDICTED PROBABILITIES __________________")
        # print(y_predicted)
        # print("  PREDICTED LABELS ------------------------")
        # print(y_pred_labels)

        # print(len(y_true), len(y_predicted), len(y_pred_labels))

        # s = torch.sum(torch.stack(y_predicted))
        # a = torch.sum(torch.stack(y_true))

    return y_true, y_predicted, y_pred_labels

def create_model(train_loader, num_epochs=2, learning_rate = 0.01):

    model = PredictorCNN()
    train(model, train_loader,num_epochs,learning_rate)

    return model











