"""
Authors : Mehdi Mitiche

This file stores the two classes of Neural Networks models : 
NNRegressor which is a model to preform regression
NNClassifier which is a model to perform classification
"""

from torch import cat, nn, no_grad, argmax
from torch.nn import Module, ModuleList, Embedding, Linear, MSELoss, BatchNorm1d, Dropout, Sequential, CrossEntropyLoss
from torch.nn.functional import log_softmax
from typing import List, Optional


class NNModel(Module):
    def __init__(self, num_cont_col: int, output_size: int, layers: List[int], activation:str, dropout: float = 0.4,
                 cat_sizes: Optional[List[int]] = None):
        """
        Creates a Neural Network model, entity embedding
        is performed on the categorical data if cat_sizes is not null

        :param num_cont_col: the number of continuous columns we have
        :param output_size: the number of nodes in the last layer of the neural network or the the number of classes
        :param layers: a list to represent the number of hidden layers and the number of units in each layer
        :param activation: the activation function to be used by the model
        :param dropout: a fraction representing the probability of dropout
        :param cat_sizes: list of integer representing the size of each categorical column
        """
        super(NNModel, self).__init__()
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

        # we initialize an empty list that will contain all the layers of our model
        all_layers = []

        # we create the different layers of our model : Linear --> ReLU --> Batch Normalization --> Dropout
        for i in layers:
            # Linear Layer
            all_layers.append(Linear(input_size, i))
            # Activation function
            all_layers.append(getattr(nn, activation)())
            # batch normalization
            all_layers.append(BatchNorm1d(i))
            # dropout layer
            all_layers.append(Dropout(dropout))
            input_size = i

        # we define the output layer
        if len(layers) == 0:
            all_layers.append(Linear(input_size, output_size))
        else:
            all_layers.append(Linear(layers[-1], output_size))

        # we save all our layers in self.layers
        self.layers = Sequential(*all_layers)

    def forward(self, x_cont, x_cat=None):
        embeddings = []
        if x_cat is not None:

            # we perform the entity embedding
            for i, e in enumerate(self.embedding_layers):
                embeddings.append(e(x_cat[:, i].long()))

            # we concatenate all the embeddings
            x = cat(embeddings, 1)

            # we concatenate categorical and continuous data after performing the  entity embedding
            x = cat([x, x_cont], 1)
        else:
            x = x_cont
        return self.layers(x)

    def predict(self, x_cont, x_cat=None):
        """
        Abstract method for model prediction function

        :param x_cont: tensor with continuous inputs
        :param x_cat: tensor with categorical ordinal encoding
        """
        raise NotImplementedError


class NNRegressor(NNModel):

    def __init__(self, num_cont_col: int, layers: List[int], activation: str, dropout:float = 0.4,
                 cat_sizes:Optional[List[int]]=None):
        """Creates a Neural Network model that perform a regression with predicting real values, entity embedding is
        performed on the data if cat_sizes is not null

        :param num_cont_col: the number of continuous columns we have
        :param layers: a list to represent the number of hidden layers and the number of units in each layer
        :param activation: the activation function to be used by the model
        :param dropout: a fraction representing the probability of dropout
        :param cat_sizes: list of integer representing the size of each categorical column
        """
        super().__init__(num_cont_col=num_cont_col, output_size=1, layers=layers, activation=activation,
                         dropout=dropout, cat_sizes=cat_sizes)

        # we define the criterion for that model
        self.criterion = MSELoss()

    def loss(self, pred, y):
        return self.criterion(pred.flatten(), y.float())

    def predict(self, x_cont, x_cat=None, **kwargs):
        """
        Returns the real-valued predictions

        :param x_cont: tensor with continuous inputs
        :param x_cat: tensor with categorical ordinal encoding
        :return: (N, 1) tensor
        """

        # We turn in eval mode
        self.eval()

        # We execute a forward pass
        with no_grad():
            output = self.forward(x_cont, x_cat)

        return output


class NNClassifier(NNModel):
    def __init__(self, num_cont_col: int, output_size: int, layers: List[int], activation: str, dropout: float = 0.4,
                 cat_sizes: Optional[List[int]]=None):
        """ Creates a Neural Network model that perform a regression With predicting real values, entity embedding is
        performed on the data if cat_sizes is not null
        :param num_cont_col: the number of continuous columns we have
        :param output_size: the number of nodes in the last layer of the neural network or the the number of classes
        :param layers: a list to represent the number of hidden layers and the number of units in each layer
        :param activation: the activation function to be used by the model
        :param dropout: a fraction representing the probability of dropout
        :param cat_sizes: list of integer representing
        the size of each categorical column
        """
        super().__init__(num_cont_col=num_cont_col, output_size=output_size, layers=layers, activation=activation,
                         dropout=dropout,
                         cat_sizes=cat_sizes)

        # we define the criterion for that model
        self.criterion = CrossEntropyLoss()

    def loss(self, pred, y):
        return self.criterion(pred, y.long())

    def predict(self, x_cont, x_cat=None, **kwargs):
        """
        Returns the log probabilities related to each class if log_prob = True,
        else it returns the classes predicted

        :param x_cont: tensor with continuous inputs
        :param x_cat: tensor with categorical ordinal encoding
        :return: (N, C) tensor
        """
        # We turn in eval mode
        self.eval()

        # We execute a forward pass
        with no_grad():
            output = self.forward(x_cont, x_cat)
            if kwargs.get("log_prob", False):
                log_soft = log_softmax(output, dim=1).float()
                return log_soft
            else:
                return argmax(output, dim=1).long()


