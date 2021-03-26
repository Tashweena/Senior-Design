from scipy.optimize import minimize
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import sys
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, RidgeCV
from sklearn.metrics import mean_squared_error
# for debugging
import time
import matplotlib.pyplot as plt

#define the sigmoid function
def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

# custom loss function
def objective_function(weight, X, Y):
  loss = -1.0 * np.matmul(sigmoid(np.matmul(X,weight)), Y.T)/(len(X))
  return loss

# log loss; Y is the true residuals [-1, 1]
def log_loss(weight, X, Y):
  # squeez Y to [0,1]
  Y = (Y + 1.0)/2.0
  y_pred = sigmoid(np.matmul(X,weight))
  y_pred = project(y_pred)
  # print(Y.T, np.log(y_pred))
  # print(Y.T @ np.log(y_pred))
  a = np.matmul(Y.T, np.log(y_pred))
  # print(a)
  b = np.matmul((1-Y).T, np.log(1-y_pred))
  
  # print('log loss y_pred', y_pred)
  # print(a)
  # print(b)
  loss = - (a + b) / len(Y)
  return loss

#make sure y_pred is in (0,1)
def project(y_pred):
  index = y_pred > (1 - 0.0001)
  y_pred[index] = y_pred[index] - y_pred[index] + 0.9999
  index = y_pred < (0.0001)
  y_pred[index] =  y_pred[index] - y_pred[index] + 0.0001
  return y_pred


#minimizing the objective loss function
def linearClassification(X, Y):
  mu, sigma = 0, 0.1
  s = np.random.normal(mu, sigma, X.shape[1])
  weight_init = np.array(s)
  weight_init = weight_init/np.linalg.norm(weight_init)
  # print(weight_init, type(weight_init))
  # print(weight_init.shape)
  # print('optimizing')
  start_time = time.time()
  # result = minimize(objective_function, weight_init, args=(X,Y), method='CG', options={'maxiter': 200})
  w, loss_lst = sgd(weight_init, X, Y, rate = 0.00001, verbose=False)
  # print("--- %s seconds ---" % (time.time() - start_time))
  return w, loss_lst

#derivative of sigmoid: df(z)/dz
def d_sigmoid(z):
  sig = sigmoid(z)
  return sig * (1 - sig)

#derivative of log loss: dL/df(x)
def d_log_loss(pred, y):
  value = - (pred - y)/((pred-1) * pred)
  return value

#back propagate for one iter
def backprop(w, x, y, rate = 0.001, verbose=False):
  y = (y + 1.0)/2
  pred = sigmoid(np.matmul(x,w))
  if pred == 0:
    pred = 0.0001
  elif pred == 1:
    pred = 0.9999
  d_loss = d_log_loss(pred, y)
  d_sig = d_sigmoid(pred)
  # dL/dw = dL/df(x) * df(x)/dz * dz/dw
  d = d_loss * d_sig * x
  w = w - rate * d 
  if verbose:
    print('pred:', pred, ' true:', y)
    # print('d_loss:', d_loss)
    # print('d_sig:', d_sig)
  return w

#stochastic gradient descent for log loss
def sgd(w, X, Y, rate = 0.001, verbose = False):
  m = len(Y)
  # print('m:', m)
  # print('Y:', type(X))
  loss_lst = []
  if verbose:
    count = 0
  for j in range(30):
    for i in np.random.permutation(m):
      # debuging, delete later:
      # if count == 3:
      #   break
      x = X[i,:]
      y = Y.iloc[i] # Y is pd.Series
      # y = Y[i]
      w = backprop(w, x, y, rate = rate, verbose=False)
      # print('w:', w)
      loss = log_loss(w, X, Y)
      loss_lst.append(loss)
      if verbose:
        #for debugging, delete later
        # print('w updated:', w)
        print('loss at count ', count, ': ', loss)
        count += 1
        # for debugging, delete later
        if loss == float('inf') or loss == np.nan:
          print('!!!! loss is inf or nans')
          break
        print('--------------------------------')
    # print(loss)
    # print('predicted residuals:', sigmoid(np.matmul(X,w)))
  
  return w, loss_lst


# inputs: the features of the dataset, the predictions from our model, and the 
# true labels of each data poinnt
# outputs: the indices from our dataset that don't predict well, given 
# the current model, and the model used to classify points that don't predict well

def learning_oracle_consistency_algorithm(x_val, y_pred, y_val):

  #find residuals, difference between predictions and true labels
  y_pred = y_pred
  y_val = y_val
  pos_resid = (y_pred - y_val)
  neg_resid =  -1 * pos_resid

  # print(pos_resid)
  # print(neg_resid)
  # print(x_val)
  #add a vector of ones to our dataset to include a bias term in our model
  # x_val = np.concatenate((x_val, np.ones((len(x_val), 1))), 1)

  beta1 = linearClassificationlinearregression(x_val,pos_resid)
  # beta2 = linearClassificationlinearregression(x_val,neg_resid)

  # #train our model with the positive residuals
  # beta1, loss_lst1 = linearClassification(x_val, pos_resid) 
  # #train our model with the negative residuals
  # beta2, loss_lst2 = linearClassification(x_val, neg_resid) 

  # return beta1,beta2
  return beta1
  # return beta1, beta2, loss_lst1, loss_lst2


def linearClassificationlinearregression(X, Y):
  # print("In the linear regression ---")
  # print(X.shape, Y.shape)
  a = 0.01
  regr = Ridge(alpha=a) 
  ### without regularization, it learns the residuals perfectly
  # linear_reg = LinearRegression().fit(X,Y) # -- not a good idea, it doesnt perform well for test
  # print("R2 score for training without ridge ", linear_reg.score(X,Y))
  regr.fit(X,Y)
  # print("R2 score for training ", regr.score(X,Y), ' penalty: ',a)
  # print(regr.intercept_)
  # print(regr.coef_.shape, np.array([regr.intercept_]).shape)
  beta = np.concatenate((regr.coef_, np.array([regr.intercept_])))
  # beta_without_ridge = np.concatenate((linear_reg.coef_, np.array([linear_reg.intercept_])))
  # alpha_val = [1e-3, 0.001,0.003,0.005,0.007,0.009,1e-1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1]
  # clf = RidgeCV(alphas=[1e-3, 1e-2,0.002,0.003, 0.004, 0.006, 1e-1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1], cv=5).fit(X, Y)
  # print(clf.alpha_, " CROSS VALIDATED R2:  ", clf.score(X,Y))

  # y_pred = regr.predict(X)
  # x_val = np.concatenate((X, np.ones((len(X), 1))), 1)
  # y_pred_other = np.matmul(x_val,beta)
  # print(y_pred)
  # print(y_pred_other)
  # print("R2 score", regr.score(X,Y))
  # # print("Coefficient of regression:", regr.coef_, regr.intercept_)
  # print("MSE ", mean_squared_error(Y,y_pred))
  # plt.figure()
  # plt.scatter(Y,y_pred)
  
  return beta
  # beta_without_ridge
  # beta


