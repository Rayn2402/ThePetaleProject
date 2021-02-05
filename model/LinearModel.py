import numpy as np


#To be converted to tensors instead of numpy arrays

class LinearRegressor():
    def __init__(self, data, cont_cols, target):
        """
        the class that will represent the analytical solution of the linear regression
        Parameters
        ----------
        data: pandas data frame
            The data frame object for the input data. It must
            contain all the features and the target column
        features: List of strings
            The names of the feature columns in the data.
        target: string
            The name of the output variable column in the data
            provided.
        """
        #we extract and read the data as torch tensors
        x, y = torch.from_numpy(data[cont_cols].values.astype(float)), torch.from_numpy(data[target].values.astype(float))
        #we add a columns containing only ones
        x = torch.cat([torch.ones(x.shape[0], 1) , x], dim=1)
        #we save the needed parameters
        self.target = target
        self.cont_cols = cont_cols
        self.b = torch.matmul(torch.matmul(torch.inverse( torch.matmul(torch.transpose(x, 0, 1), x)), torch.transpose(x, 0, 1)), y )
    def predict(self, x_data):
        """
        function that returns the predictions of a given data
        """
        x_data = torch.from_numpy(x_data[self.cont_cols].values.astype(float))
        x_data = torch.cat([torch.ones(x_data.shape[0], 1), x_data], dim=1)
        return  torch.matmul(x, self.b)
    def test(self, data):
        """
        function that tests the model by returning the error of the prediction of a given data
        """
        #we extract the data needed
        x, y = torch.from_numpy(data[self.cont_cols].values.astype(float)), torch.from_numpy(data[self.target].values.astype(float))
        #we add a column comtaining omly ones
        x = torch.cat([torch.ones(x.shape[0], 1) , x], dim=1)
        #we make our prediction
        y_pred = torch.matmul(x, self.b)
        #we return the metric
        return self.rmse(y_pred, y)
    def get_solution(self):
        """
        function that returns the solution of the linear regression
        """
        return self.b
    def rmse(self, y_pred, y):
        """
        functiom that returns the Root Mean Squared Error
        """
        #we caclulate the root mean square error
        return torch.sqrt(torch.mean(torch.square(y_pred - y))).item()


class GDLinearRegressor():
    def __init__(self):
        