"""
Authors : Mehdi Mitiche

This file stores the two classes of the linear regression models : 
LinearRegressor which is the class that will represent the analytical solution and
GDLinearRegressor which is the class that will represent the model of the linear regression with gradient descent
"""

from torch import randn, matmul, cat, inverse, transpose, eye, tensor
from torch.nn import Module, ModuleList, Embedding, Linear, MSELoss


class LinearRegression:
    """
    Linear regression using the analytical solution with MSE loss
    """
    def __init__(self, input_size: int, beta: float = 0):
        """
        Sets the weights tensor (input_size, 1) and the L2 penalty

        Args:
            input_size: number of column in the input matrix
            beta: L2 penalty coefficient
        """

        # We set the private attributes
        self.__w = randn(input_size, 1)
        self.__beta = beta

    @property
    def w(self):
        return self.__w

    @property
    def beta(self):
        return self.__beta

    def fit(self, x: tensor, y: tensor) -> None:
        """
        Computes the optimal weights using the analytical solution

        Args:
            x: input tensor (N, input_size)
            y: targets (N, 1)

        Returns: None

        """
        # we find weights using the analytical solution formula of the linear regression
        self.__w = matmul(matmul(inverse((matmul(transpose(x, 0, 1), x) +
                                          eye(x.shape[1]) * self.__beta)), transpose(x, 0, 1)), y)

    def loss(self, x: tensor, y: tensor) -> float:
        """
        Returns the MSE loss associated to the input and the targets

        Args:
            x: input tensor (N, input_size)
            y: targets (N, 1)

        Returns: MSE loss

        """
        return ((self.predict(x) - y) ** 2).mean().item()

    def predict(self, x: tensor) -> tensor:
        """
        Multiplies the input tensor with the weight tensor
        Args:
            x: input tensor (N, input_size)

        Returns: tensor with predictions

        """
        return matmul(x, self.__w)
