"""
Authors : Mehdi Mitiche

This file stores the two classes of the linear regression models : 
LinearRegressor which is the class that will represent the analytical solution and
GDLinearRegressor which is the class that will represent the model of the linear regression with gradient descent
"""

from torch import randn, matmul, cat, inverse, transpose, eye
from torch.nn import Module, ModuleList, Embedding, Linear, MSELoss


class LinearRegressor:

    def __init__(self, input_size, regularization=False, lambda_value=None):
        """
        Creates a model that give us the analytical solution of the  linear regression 
        :param input_size: the number of features we have
        :param regularization: (boolean) Specify if we want to use regularization (True) or not (False)
        :param lambda_value: (number) The lambda value used in the regularization
        """

        # We check if the regularization_value is specified if the regularization is activated
        assert regularization is False or lambda_value is not None, "regularization value must be specified"

        # we initialize the weights with random numbers
        self.W = randn(input_size, 1)
        self.regularization = regularization
        self.lambda_value = lambda_value

    def train(self, x, y):
        # we find weights using the analytical solution formula of the linear regression
        if self.regularization is False:
            self.W = matmul(matmul(inverse(matmul(transpose(x, 0, 1), x)), transpose(x, 0, 1)), y)
        else:
            self.W = matmul(matmul(inverse((matmul(transpose(x, 0, 1), x) +
                                            eye(x.shape[1]) * self.lambda_value)), transpose(x, 0, 1)), y)

    def predict(self, x):
        """
        function that returns the predictions of a given data
        """
        return matmul(x, self.W)

    def loss(self, x, target):
        """
        function that evaluates the model by returning the error of the prediction of a given data
        """
        return ((self.predict(x) - target) ** 2).mean().item()


class GDLinearRegressor(Module):
    def __init__(self, num_cont_col, cat_sizes=None):
        """
        Creates a model that perform a linear regression with gradient descent, entity embedding
        is performed on the data if cat_sizes is not null

        :param cat_sizes: list of integer representing the size of each categorical column
        :param num_cont_col: the number of continuous columns we have
        """
        super(GDLinearRegressor, self).__init__()
        if cat_sizes is not None:

            # we generate the embedding sizes ( this part will be optimized )
            embedding_sizes = [(cat_size, min(50, (cat_size + 1) // 2)) for cat_size in cat_sizes]

            # we create the Embeddings layers
            self.embedding_layers = ModuleList([Embedding(num_embedding, embedding_dim) for
                                                num_embedding, embedding_dim in embedding_sizes])

            # we get the number of our categorical data after the embedding ( we sum the embeddings dims)
            num_cat_col = sum((embedding_dim for num_embedding, embedding_dim in embedding_sizes))

            # the number of entries to our linear layer
            input_size = num_cat_col + num_cont_col
        else:
            # the number of entries to our linear layer
            input_size = num_cont_col

        # we define our linear layer
        self.linear = Linear(input_size, 1)
        # we define the criterion for that model
        self.criterion = MSELoss()

    def forward(self, x_cont, x_cat=None):

        embeddings = []
        if x_cat is not None:

            # we perform the entity embedding
            for i, e in enumerate(self.embedding_layers):
                embeddings.append(e(x_cat[:, i]))

            # we concatenate all the embeddings
            x = cat(embeddings, 1)

            # we concatenate categorical and continuous data after performing the  entity embedding
            x = cat([x, x_cont], 1)
        else:
            x = x_cont
        return self.linear(x)

    def loss(self, x_cont, x_cat, target):
        return ((self(x_cont.float(), x_cat).squeeze() - target) ** 2).mean().item()
