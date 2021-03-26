import pandas as pd
import numpy as np
import pdb
class Predictor:
  def __init__(self, initial_predictor,x_train,y_train):
    self.initial_predictor = initial_predictor
    if initial_predictor != None:
      self.initial_predictor.fit(x_train,y_train)
    else:
      self.val = 0.5
    self.lbda_group_list = []

  def predict_y(self, x_test, x_test_sub, dummy):
    if dummy:
      # print("The dummy one")
      y_pred = [self.val] * len(x_test)
      # y_pred = x_test[:,-1].values
      y_pred = np.array(y_pred)
    else:
      y_pred = self.initial_predictor.predict(x_test)
      

    #ADDED below line - REMOVE LATER AFTER LINEAR IS DONE 
    # y_pred = np.squeeze(y_pred)
    # print(y_pred)
    if y_pred.ndim > 1:
      y_pred = y_pred.reshape(-1)
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.apply(lambda i : 0 if i <= 0 else min(i,1))
    

    for func, val in self.lbda_group_list:
      indices = func(x_test_sub, y_pred)
      # print("Going through this ", val, "sum ", sum(indices))
      # Below line is for linear data
      # y_pred = y_pred.values.reshape((y_pred.shape[0], 1))
      
      y_pred[indices] += val

      # projection = lambda i : 0 if i <= 0 else min(i, 1)
      # y_pred = np.array([projection(ele) for ele in y_pred])
      # Below line is for linear data
      # y_pred = pd.Series(np.squeeze(y_pred))
      y_pred[indices] = y_pred[indices].apply(lambda i : 0 if i <= 0 else min(i,1))
        # print("The NEW mean is ")
        # print(np.mean(y_pred[indices]))
        ## DO THE PROJECTION HERE

    return y_pred

  def predict_y_var(self, x_test, x_test_sub, batch_variance, calibrated_means, dummy):
            y_pred = [0.5] * len(x_test)
            #print("length of x test", len(x_test))
            y_pred = pd.Series(y_pred)
            #print("y_pred", y_pred)
            for func, val in self.lbda_group_list:
                indices = func(x_test_sub, calibrated_means, y_pred)
                #print("indices in predictor", sum(indices))
                y_pred[indices] += val
                #print("y_pred in for loop ", y_pred)

                y_pred[indices] = y_pred[indices].apply(lambda i : 0 if i <= 0 else min(i,1))
            #print("y_pred after for loop ", y_pred)

            return y_pred

  def predict_y_var_linear_data(self, x_test, x_test_sub, batch_variance, calibrated_means, dummy):
          y_pred = [0.5] * len(x_test)
          y_pred = np.squeeze(y_pred)
          y_pred = pd.Series(y_pred)
          for func, val in self.lbda_group_list:
              indices = func(x_test_sub, calibrated_means, y_pred)
              y_pred = y_pred.values.reshape((y_pred.shape[0], 1))
              print("y_pred before", str(y_pred))
              y_pred[indices] += val
              print("y_pred after", str(y_pred))

              projection = lambda i : 0 if i <= 0 else min(i, 1)
              y_pred = np.array([projection(ele) for ele in y_pred])
              y_pred = pd.Series(np.squeeze(y_pred))
          return y_pred
  # amount is eta*lbda
  def update(self, sub_picker, amount):
    self.lbda_group_list.append((sub_picker,amount))
    # print(self.lbda_group_list)


#####################################################################


# import pandas as pd
# import numpy as np
# class Predictor:
#   def __init__(self, initial_predictor,x_train,y_train):

#     self.initial_predictor = initial_predictor
#     self.initial_predictor.fit(x_train,y_train)
#     self.lbda_group_list = []   

#   def predict_y(self, x_test, x_test_sub):
#     y_pred = self.initial_predictor.predict(x_test)
#     y_pred = pd.Series(y_pred)
#     for func, val in self.lbda_group_list:
#       indices = func(x_test_sub, y_pred)
#       y_pred[indices] += val
#       y_pred[indices] = y_pred[indices].apply(lambda i : 0 if i <= 0 else min(i,1))
#       # print("The NEW mean is ")
#       # print(np.mean(y_pred[indices]))
#       ## DO THE PROJECTION HERE

#     return y_pred

#   # amount is eta*lbda
#   def update(self, sub_picker, amount):
#     # print("UPDATE IS BEING CALLED")
#     self.lbda_group_list.append((sub_picker,amount))
#     # print(self.lbda_group_list)


