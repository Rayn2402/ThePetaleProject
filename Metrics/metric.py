import numpy as np

def RMSE(y_pred, y):
  return np.sqrt((np.square(y_pred - y).sum()) /y.shape[0])