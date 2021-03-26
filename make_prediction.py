import numpy as np
import pandas as pd
def predict(y_pred, x_test, update_list, store_changes=True, sub_indices=None):
  y_pred_fix = y_pred.copy()
  if store_changes and (sub_indices is not None):
    T = len(update_list)
    initial_subgroup_pred_mean = [np.mean(y_pred_fix[sub_indices])]*T
  else:
    initial_subgroup_pred_mean = []
  i = 0
  # print("the mean before is ", np.mean(y_pred_fix[sub_indices]))
  for func, val in update_list:
    indices = func(x_test, y_pred_fix)
    
    y_pred_fix[indices] += val
    if store_changes:
      initial_subgroup_pred_mean[i] = np.mean(y_pred_fix[sub_indices])
      i += 1
  return y_pred_fix, initial_subgroup_pred_mean

def predict_linear(y_pred, x_test, update_list, store_changes=True, sub_indices=None):
  y_pred_fix = y_pred.copy()
  y_pred_fix = y_pred_fix.reshape(y_pred_fix.shape[0],1)
  if store_changes and (sub_indices is not None):
    T = len(update_list)
    initial_subgroup_pred_mean = [np.mean(y_pred_fix[sub_indices])]*T
  else:
    initial_subgroup_pred_mean = []
  i = 0
  # print("the mean before is ", np.mean(y_pred_fix[sub_indices]))
  for func, val in update_list:
    y_pred_fix = pd.Series(np.squeeze(y_pred_fix))
    indices = func(x_test, y_pred_fix)
    
    y_pred_fix = y_pred_fix.values.reshape(y_pred_fix.shape[0],1)
    y_pred_fix[indices] += val
    if store_changes:
      initial_subgroup_pred_mean[i] = np.mean(y_pred_fix[sub_indices])
      i += 1
  return y_pred_fix, initial_subgroup_pred_mean

def predict_variance(y_pred, x_test, calibrated_means, update_list, store_changes=True, sub_indices=None):
  y_pred_fix = y_pred.copy()
  if store_changes and (True in sub_indices):
    T = len(update_list)
    initial_subgroup_pred_mean = [np.mean(y_pred_fix[sub_indices])]*T
  else:
    initial_subgroup_pred_mean = []
  i = 0
  # print("the mean before is ", np.mean(y_pred_fix[sub_indices]))
  for func, val in update_list:
    indices = func(x_test, calibrated_means, y_pred_fix)
    y_pred_fix[indices] += val
    print(indices, val)
    y_pred_fix[indices] = y_pred_fix[indices].apply(lambda i : 0 if i <= 0 else min(i,1))
    if store_changes and (len(initial_subgroup_pred_mean) > 0):
      initial_subgroup_pred_mean[i] = np.mean(y_pred_fix[sub_indices])
      i += 1
  #print("the fixed y_pred is :", y_pred_fix)
  return y_pred_fix, initial_subgroup_pred_mean

def predict_linear_variance(y_pred, x_test, calibrated_means, update_list, store_changes=True, sub_indices=None):
  y_pred_fix = y_pred.copy()
  y_pred_fix = y_pred_fix.reshape(y_pred_fix.shape[0],1)
  if store_changes and (sub_indices is not None):
    T = len(update_list)
    initial_subgroup_pred_mean = [np.mean(y_pred_fix[sub_indices])]*T
  else:
    initial_subgroup_pred_mean = []
  i = 0
  # print("the mean before is ", np.mean(y_pred_fix[sub_indices]))
  for func, val in update_list:
    y_pred_fix = pd.Series(np.squeeze(y_pred_fix))
    indices = func(x_test, calibrated_means, y_pred_fix)
    
    y_pred_fix = y_pred_fix.values.reshape(y_pred_fix.shape[0],1)
    y_pred_fix[indices] += val
    if store_changes:
      initial_subgroup_pred_mean[i] = np.mean(y_pred_fix[sub_indices])
      i += 1
  return y_pred_fix, initial_subgroup_pred_mean