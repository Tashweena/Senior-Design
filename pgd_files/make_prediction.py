import numpy as np
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