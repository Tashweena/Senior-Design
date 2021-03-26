import torch
import numpy as np
class dummyModel():
  def __init__(self):
    pass

  def predict(self, images):
    images = torch.flatten(images, start_dim=1)
    first_col = images[:,0]
    prediction = [0.5 for i in first_col]
    # prediction = (first_col + 1)/2
    ### Real Dummy predictor that predicts 0.5 for every image
    return np.array(prediction)
    # prediction.numpy()

  def fit(self, x_train, y_train):
    return

