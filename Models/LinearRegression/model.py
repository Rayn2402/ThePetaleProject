import numpy as np

class AnalyticalLinearRegression():
    def __init__(self, data, features, target):
        """
        class that will represent the analytical solution of the linear regression
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
        #we extract and read the data as numpy arrays **TO BE CONVERTED TO TENSORS
        y = data[target].values.astype(float)
        x = data[features].values.astype(float)
        #we add a columns containing only ones
        x = np.concatenate((np.ones((x.shape[0],1)), x),axis = 1)
        #we save the needed parameters
        self.target = target
        self.features = features
        self.b = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

    def predict(self, x_data):
        """
        function that returns the predictions of a given data
        """
        x_data = x_data[self.features].values.astype(float)
        x_data = np.concatenate((np.ones((x_data.shape[0],1)), x_data),axis = 1)
        return x_data.dot(self.b)

    def test(self, data):
        """
        function that tests the model by returning the error of the prediction of a given data
        """
        #we extract the data needed
        y = data[self.target].values.astype(float)
        x = data[self.features].values.astype(float)
        #we add a column comtaining omly ones
        x = np.concatenate((np.ones((x.shape[0],1)), x),axis = 1)
        #we make our prediction
        y_pred = x.dot(self.b)
        #we return the metric
        return self.rmse(y_pred, y)
    
    def get_solution(self):
        """
        function that returns the solution of the linear regression
        """
        return self.b

    def rmse(self, y_pred, y):
        """
        functiom that return the Root Mean Squared Error
        """
        return np.sqrt((np.square(y_pred - y).sum()) /y.shape[0])

