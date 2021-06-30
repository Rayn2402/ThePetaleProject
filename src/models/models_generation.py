"""
Authors : Mehdi Mitiche

File that contains the class that will be responsible of generating the model when we tune the hyper parameters

"""

from src.models.nn_models import NNModel
from sklearn.linear_model import ElasticNet
from typing import Callable, List, Optional, Union


def build_elasticnet(alpha: float, beta: float):
    """
    Creates an ElasticNet model from sklearn

    Args:
        alpha: L1 penalty coefficient
        beta: L2 penalty coefficient

    Returns: elasticnet model
    """
    l1_ratio = alpha / (alpha + beta)
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=10000)


class NNModelGenerator:
    """
    Object responsible of generating nn model according to a set of hyperparameter
    """
    def __init__(self, model_class: Callable, num_cont_col: int,
                 cat_sizes: Optional[List[int]] = None, output_size: Optional[int] = None):
        """
        Sets private attributes

        Args:
            model_class: constructor of NNClassifier or NNRegression
            num_cont_col: number of continuous columns
            cat_sizes: list of integer representing the size of each categorical column
            output_size: number of nodes in the last layer of the neural network/the the number of classes
        """
        # Private attributes
        self.__cat_sizes = cat_sizes
        self.__model = model_class
        self.__num_cont_col = num_cont_col
        self.__output_size = output_size

    def __call__(self, layers: List[int], dropout: float, activation: str,
                 alpha: float = 0, beta: float = 0) -> NNModel:
        """
        Generates a neural network model associated to the given set of hyperparameters

        Args:
            layers: list with number of nodes for each hidden layer
            dropout: probability of dropout (0 < p < 1)
            activation: activation function to be used by the model (ex. "ReLU")
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient

        Returns: neural network

        """
        return self.__model(num_cont_col=self.__num_cont_col, cat_sizes=self.__cat_sizes,
                            output_size=self.__output_size, layers=layers, dropout=dropout, activation=activation)
