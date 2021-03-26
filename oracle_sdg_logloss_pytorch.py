import numpy as np
import torch

## Pytorch implementation of sgd using log loss 
# https://m-alcu.github.io/blog/2018/02/10/logit-pytorch/

#define the sigmoid function
def sigmoid(z):
  return 1.0 / (1 + torch.exp(-z))

## make the prediction for the residual
def model(weight,X):
  # print(X.shape, weight.shape)
  # print(type(X),type(weight))
  pred = sigmoid(X @ weight)
  ## project if necessary
  for i in range(len(pred)):
    if pred[i] == 0:
      pred[i] = 0.0001
    elif pred[i] == 1:
      pred[i] = 0.9999
  
  return pred

# log loss; Y is the true residuals [-1, 1]
def log_loss(Y,y_pred):
  # squeez Y to [0,1]
  Y = (Y + 1.0)/2.0
  # print("squeezed residuals for Y ", Y)
  # print("True residuals after squeezing ", Y)
  # print("Y -- true residuals", type(Y))
  Y = np.array(Y)
  Y_tensor = torch.from_numpy(Y.T)
  Y_tensor = torch.autograd.Variable(Y_tensor, requires_grad=True)
  a = (Y_tensor @ torch.log(y_pred))
  b = ((1-Y_tensor) @ torch.log(1-y_pred))

  loss = - (a + b) / len(Y)
  return loss

def sgd(w, X, Y, rate = 0.001, epochs = 20, verbose = False):
  w = torch.from_numpy(w)
  w = torch.autograd.Variable(w, requires_grad=True)
  weight_tensor = {'params' : w }
  # print("Weight_tensor ", weight_tensor)
  # print("X ", X)
  optimizer = torch.optim.SGD([weight_tensor], lr=rate, momentum=0.9)
  loss_lst = []
  for i in range(epochs):
    y_pred = model(w,X)
    # print("residuals predicted", y_pred)
    loss = log_loss(Y,y_pred)
    # print("loss is ", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_lst.append(loss.detach())
  return w.detach().numpy(),loss_lst


#minimizing the loss function -- log loss
def linearClassification(X, Y):
  mu, sigma = 0, 0.1
  s = np.random.normal(mu, sigma, X.shape[1])
  weight_init = np.array(s)
  weight_init = weight_init/np.linalg.norm(weight_init)
  # print("initial weight_vector ", weight_init, type(weight_init))
  # print("weight_vector shape ", weight_init.shape)
  w, loss_lst = sgd(weight_init, X, Y, rate = 0.001, epochs=50, verbose=False)
  return w, loss_lst

def learning_oracle_consistency_algorithm(x_val, y_pred, y_val):
  #find residuals, difference between predictions and true labels
  y_pred = y_pred
  y_val = y_val
  pos_resid = (y_pred - y_val)
  neg_resid =  -1 * pos_resid

  #add a vector of ones to our dataset to include a bias term in our model
  x_val = np.concatenate((x_val, np.ones((len(x_val), 1))), 1)
  x_val = torch.from_numpy(x_val)
  #train our model with the positive residuals
  # print("pos_resid ", pos_resid)
  beta1, loss_lst1 = linearClassification(x_val, pos_resid) 
  #train our model with the negative residuals
  beta2, loss_lst2 = linearClassification(x_val, neg_resid) 
  return beta1, beta2, loss_lst1, loss_lst2
  # return beta1, loss_lst1
  