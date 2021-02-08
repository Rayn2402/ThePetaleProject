"""
Authors : Mehdi Mitiche

This file stores the two classes of the linear regression models : 
LinearRegressor wich is the class that will represent the analytical solution and GDLinearRegressor wich is the class that represent the model of linear regression yith gradiant decent
"""

import  torch

class LinearRegressor():
    def __init__(self, input_size):
        #we intialize the weights with random numbers
        self.W = torch.randn(input_size, 1)
    def train(self, x, y):
        #we find weights using the analytical solution formula of the linear regression
        self.W = torch.matmul(torch.matmul(torch.inverse( torch.matmul(torch.transpose(x, 0, 1), x)), torch.transpose(x, 0, 1)), y )
    def predict(self, x):
        """
        function that returns the predictions of a given data
        """
        return  torch.matmul(x, self.W)
    def loss(self, x, target):
        """
        function that evaluates the model by returning the error of the prediction of a given data
        """
        return ((self.predict(x).unsqueeze(dim=0) - target)**2).mean().item()