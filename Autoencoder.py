import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderCnn(nn.Module):
    """ Autoencoder for CIFAR10
    
    Creates a 100-dim representation of a 32*32*3 image
    
    Adapted model architecture from 
    https://github.com/jellycsc/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
   
    """
    def __init__(self):
        super(AutoencoderCnn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),           
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),          
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),          
            nn.ReLU(),
			nn.Conv2d(48, 96, 4, stride=2, padding=1),    
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(384,100)
        )
        self.decoder = nn.Sequential(
            nn.Linear(100,384),
            nn.Unflatten(-1,(96,2,2)),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), 
            nn.ReLU(),
		    nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(data_loader, epochs=125):
    """ This functions trains an autoencoder for CIFAR10 dataset
    to create a low dimensional representation of the image for the learning oracle.

    Inputs: 
    
    data_loader - DataLoader object to train the autoencoder
    
    epochs - (int) num epochs to train for
   
    Outputs: 
    
    model_cnn - a trained autoencoder
    
    loss_arr - list, reconstruction loss for each epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cnn = AutoencoderCnn().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_cnn.parameters())
    loss_arr = []
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in data_loader:
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            # compute reconstructions
            encoded, decoded = model_cnn(batch_features)
            # compute training reconstruction loss
            train_loss = criterion(decoded, batch_features)
            # compute accumulated gradients
            train_loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(data_loader)
        loss_arr.append(loss)
        if epoch % 10 == 0:
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    
    return model_cnn, loss_arr