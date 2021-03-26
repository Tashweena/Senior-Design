from scipy.optimize import minimize
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import sys
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression

#define the sigmoid function
def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

# custom loss function
def objective_function(weight, X, Y):
  loss = -1.0 * np.matmul(sigmoid(np.matmul(X,weight)), Y.T)/(len(X))
  return loss

#minimizing the objective loss function
def linearClassification(X, Y):
  mu, sigma = 0, 0.1
  s = np.random.normal(mu, sigma, X.shape[1])
  weight_init = np.array(s)
  weight_init = weight_init/np.linalg.norm(weight_init)
  # print(weight_init.shape)
  print('optimizing')
  result = minimize(objective_function, weight_init, args=(X,Y), method='CG', options={'maxiter': 800})
  print('done optimizing')
  return result.x


# inputs: the features of the dataset, the predictions from our model, and the 
# true labels of each data poinnt
# outputs: the indices from our dataset that don't predict well, given 
# the current model, and the model used to classify points that don't predict well

def learning_oracle_consistency_algorithm(x_val, y_pred, y_val):

  #find residuals, difference between predictions and true labels
  y_pred = y_pred
  y_val = y_val
  pos_resid = (y_pred - y_val)
  neg_resid =  -1 * (pos_resid)
  pos_resid = (pos_resid + 1)/2
  neg_resid = (neg_resid + 1)/2


  #add a vector of ones to our dataset to include a bias term in our model
  x_val = np.concatenate((x_val, np.ones((len(x_val), 1))), 1)
  #train our model with the positive residuals
  print("Beta1 - positive")
  beta1 = linearClassification(x_val, pos_resid) 
  #train our model with the negative residuals
  print("Beta2 - Negative")
  beta2 = linearClassification(x_val, neg_resid) 
  return beta1, beta2


