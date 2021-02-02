import sys
import numpy as np
sys.path.append("../..")
from Metrics.metric import RMSE

class AnalyticalLinearRegression():
    def __init__(self, df, features, dep_var):

        x_train = df[features].values.astype(float)
        y_train = df[dep_var].values.astype(float)

        self.b = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
    
    def test(self, x, y):
        y_pred = x.dot(self.b)
        return RMSE(y_pred, y)
    
    def predict(self, x):
        return x.dot(self.b)

