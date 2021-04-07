import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,mean_squared_error, confusion_matrix

class PredictorCnn(nn.Module):
    def __init__(self):
        super(PredictorCnn, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.fc1 = nn.Linear(8*8*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.fc4 = nn.Linear(10,1)
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(self.fc4(x))
        return x
    
    # dummy function 
    def fit(self, x_train, y_train):
        return
       
    def predict(self, data):
        y_pred = self.forward(data)
        y_pred = y_pred.detach().numpy()
        y_pred = y_pred.flatten()
        return y_pred
   
def train(model, data_loader, num_epochs, learning_rate):
    """ Train the cnn"""
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = torch.autograd.Variable(images.float())
            labels = torch.autograd.Variable(labels)
            labels = np.reshape(labels, (-1,1))
            optimizer.zero_grad()
            outputs = model(images.float())
            # convert original labels to animal labels
            # 1 - animal, 0 otherwise
            animal_label = labels.data.tolist()
            animal_label = [[1] if i[0] in [2,3,4,5,6,7] else [0] for i in animal_label]
            animal_label = torch.Tensor(animal_label)

            loss =  F.binary_cross_entropy(outputs, animal_label)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                # evaluate the model
                y_true, y_predicted, y_pred_labels = evaluate(model, data_loader)
                print(f'Epoch : {epoch}/{num_epochs}, '
                      f'Iteration : {i}/{len(data_loader)},  '
                      f'Loss: {loss.item():.4f},',
                      f'Mean Squared Error: {mean_squared_error(y_true, y_predicted):.4f},',
                      f'Train Accuracy: {100.* accuracy_score(y_true, y_pred_labels):.4f},')

def evaluate(model, data_loader):
    """ Evaluate the model
    
    Inputs: 
    
    model - CNN model
    
    data_loader - DataLoader object(Train or Val)

    Outputs:
    
    y_true - true fly labels(0,1)
   
    y_predicted - probabilities predicted by the model
    
    y_pred_labels - predicted fly labels(0,1)
    """
    model.eval()
    y_true = []
    y_predicted = []
    y_pred_labels = []
  
    for images, labels in data_loader:
        images = torch.autograd.Variable(images.float())
        labels = np.reshape(labels, (-1,1))
        animal_label = labels.data.tolist()
        animal_label = [[1] if i[0] in [2,3,4,5,6,7] else [0] for i in animal_label]

        # predicted probabilities
        outputs = model(images) 
        # convert to animal labels 0 and 1 
        pred = outputs.data.tolist()
        pred = [[1] if e[0] >= 0.5 else [0] for e in pred]
        pred = torch.Tensor(pred)

        y_true.extend(animal_label)
        y_predicted.extend(outputs)
        y_pred_labels.extend(pred)
    
    if verbose == True:
        print("True fly labels ----------", y_true)
        print("Predicted probabilities ----------", y_predicted)
        print("Predicted fly labels ----------", y_pred_labels)
        
    return y_true, y_predicted, y_pred_labels

def create_model(train_loader, num_epochs=2, learning_rate=0.01):
    """ This functions created a convolutional neural net for CIFAR10.
    It predicts whether an object is an animal or not.
    
    Inputs:
    
    train_loader - DataLoader
    
    Num_epochs - how many iterations
    
    learning rate - default 0.01

    Outputs: a trained CNN
    """
    model = PredictorCnn()
    train(model, train_loader,num_epochs,learning_rate)
    return model

def load_model(path):
    """ This functions loads a pre trained convolutional neural net for CIFAR10.
    It predicts whether an object is an animal or not.

    Inputs: path of model (str)
   
    Outputs: a trained CNN
    """
    model = PredictorCnn()
    model.load_state_dict(torch.load(path))
    return model