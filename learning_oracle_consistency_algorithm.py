from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import sys
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from scipy.optimize import minimize
import numpy as np
import sklearn

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))


def objective_function(weight, X, Y):
  loss = -1.0 * np.matmul(sigmoid(np.matmul(X,weight)), Y.T)/(len(X))
  return loss

def linearClassification(X, Y):
  mu, sigma = 0, 0.1
  s = np.random.normal(mu, sigma, X.shape[1])
  weight_init = np.array(s)
  weight_init = weight_init/np.linalg.norm(weight_init)
  result = minimize(objective_function, weight_init, args=(X,Y), method='CG', options={'maxiter': 800})
  return result.x

def learning_oracle_consistency_algorithm(x_test, y_pred, y_test, alpha, delta):
  y_pred = y_pred.to_numpy()
  y_test = y_test.to_numpy()
  pos_resid = (y_pred - y_test)
  neg_resid =  -1 * pos_resid
  x_test = np.concatenate((x_test, np.ones((len(x_test), 1))), 1)

  beta1 = linearClassification(x_test, pos_resid) 
  y_test_resid_pos = sigmoid(np.matmul(x_test,beta1))

  beta2 = linearClassification(x_test, neg_resid) 
  y_test_resid_neg = sigmoid(np.matmul(x_test,beta2))

  pos_high_error_set = []
  neg_high_error_set = []

  for idx in range(len(y_test_resid_pos)):
    if (y_test_resid_pos[idx] > .5):
      pos_high_error_set.append(x_test[idx])
    if (y_test_resid_neg[idx] > .5):
      neg_high_error_set.append(x_test[idx])

  return pos_high_error_set, neg_high_error_set
