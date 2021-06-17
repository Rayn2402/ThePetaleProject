"""
Authors : Mehdi Mitiche

File that contains the class that will be responsible of generating the model when we tune the hyper parameters

"""

from sklearn.linear_model import ElasticNet


class ElasticNetGenerator:
    """
    Object responsible of generating ElasticNet related to a set of hps
    """
    def __call__(self, alpha: float, beta: float):
        """
        Creates an ElasticNet model from sklearn

        Args:
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient

        Returns: elasticnet model
        """
        l1_ratio = alpha / (alpha + beta)
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=5000)


class NNModelGenerator:
    def __init__(self, model_class, num_cont_col, cat_sizes=None, output_size=None):
        """
        Class that will be responsible of generating the model
        
        :param model_class: Class of the model we want to use
        :param num_cont_col: Number of continuous columns we have
        :param cat_sizes: List of integer representing the size of each categorical column
        :param output_size: Number of nodes in the last layer of the neural network or the the number of classes
        """
        self.modelClass = model_class
        self.num_cont_col = num_cont_col
        self.cat_sizes = cat_sizes
        self.output_size = output_size

    def __call__(self, layers, dropout, activation):
        """
        The method to call to generate the NN model

        :param layers: List to represent the number of hidden layers and the number of units in each layer
        :param dropout: Probability of dropout (0 < p < 1)
        :param activation: Activation function to be used by the model

        :return: Neural network model
        """
        if self.output_size is None:
            # the case when the model doesn't need the parameter output size like the model NNRegressor
            return self.modelClass(num_cont_col=self.num_cont_col, cat_sizes=self.cat_sizes, layers=layers,
                                   dropout=dropout, activation=activation)
        else:
            # the case when the model need the parameter output size
            return self.modelClass(num_cont_col=self.num_cont_col, cat_sizes=self.cat_sizes,
                                   output_size=self.output_size, layers=layers, dropout=dropout, activation=activation)
