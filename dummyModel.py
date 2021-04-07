import torch
import numpy as np
class dummyModel():
  """
  A class that represent a dummy model with custom output.
  Used for image (CIFAR10) dataset
  """
  def __init__(self):
    pass

  def predict(self, images):
    """
    Take in images and always predict 0.5
    """
    images = torch.flatten(images, start_dim=1)
    first_col = images[:,0]
    prediction = [0.5 for i in first_col]
    # prediction = (first_col + 1)/2
    ### Real Dummy predictor that predicts 0.5 for every image
    return np.array(prediction)
    # prediction.numpy()

  def fit(self, x_train, y_train):
    """
    A dummy method that does nothing.
    Created because predictor_func calls it.
    """
    return

