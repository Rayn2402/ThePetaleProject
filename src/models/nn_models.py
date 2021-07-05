"""
Authors : Mehdi Mitiche
          Nicolas Raymond

This file stores the two classes of Neural Networks models : 
    - NNRegression which is a model to preform regression
    - NNClassifier which is a model to perform classification
"""

from abc import ABC, abstractmethod
from torch import cat, nn, no_grad, argmax, tensor, mean, zeros_like, flatten
from torch.nn import Module, ModuleList, Embedding,\
    Linear, MSELoss, BatchNorm1d, Dropout, Sequential, CrossEntropyLoss
from torch.nn.functional import log_softmax, l1_loss, mse_loss
from typing import Callable, List, Optional


class NNModel(ABC, Module):
    """
    Neural Network model with entity embedding
    """
    def __init__(self, output_size: int, layers: List[int], activation: str, criterion: Callable,
                 dropout: float = 0, alpha: float = 0, beta: float = 0,  num_cont_col: Optional[int] = None, cat_sizes: Optional[List[int]] = None):

        """
        Builds the layers of the model and sets the criterion

        Args:
            output_size: the number of nodes in the last layer of the neural network

            layers: list with number of units in each hidden layer
            criterion: loss function of our model
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns
            (equal to number of class in the case of classification)
            cat_sizes: list of integer representing the size of each categorical column
        """
        assert num_cont_col is not None or cat_sizes is not None, "There must be continuous columns" \
                                                                  " or categorical columns"

        # We save penalty coefficients
        self._alpha = alpha
        self._beta = beta

        # We call parents' constructors
        Module.__init__(self)
        ABC.__init__(self)

        # We set criterion
        self._criterion = criterion

        # We initialize the input_size
        input_size = num_cont_col if num_cont_col is not None else 0

        # We set the embedding layers
        if cat_sizes is not None:

            # We generate the embedding sizes
            embedding_sizes = [(cat_size, cat_size) for cat_size in cat_sizes]

            # We create the embedding layers
            self._embedding_layers = ModuleList([Embedding(num_embedding, embedding_dim) for
                                                 num_embedding, embedding_dim in embedding_sizes])
            # We sum the length of all embeddings
            input_size += sum(cat_sizes)

        # We initialize an empty list that will contain all the layers of our model
        all_layers = []

        # We create the different layers of our model : Linear --> Activation --> Batch Normalization --> Dropout
        for i in layers:
            all_layers.append(Linear(input_size, i))
            all_layers.append(getattr(nn, activation)())
            all_layers.append(BatchNorm1d(i))
            all_layers.append(Dropout(dropout))
            input_size = i

        # We define the output layer
        if len(layers) == 0:
            all_layers.append(Linear(input_size, output_size))
        else:
            all_layers.append(Linear(layers[-1], output_size))

        # We save all our layers in self.layers
        self._layers = Sequential(*all_layers)

    def forward(self, x_cont: Optional[tensor], x_cat: Optional[tensor]) -> tensor:
        """
        Executes the forward pass

        Args:
            x_cont: tensor with continuous inputs
            x_cat: tensor with categorical ordinal encodings

        Returns: tensor with values of the node from the last layer

        """

        # We initialize list of tensors to concatenate
        x = [x_cont] if x_cont is not None else []

        if x_cat is not None:

            embeddings = []

            # We perform entity embeddings
            for i, e in enumerate(self._embedding_layers):
                embeddings.append(e(x_cat[:, i].long()))

            # We concatenate all the embeddings
            x.append(cat(embeddings, 1))

        # We concatenate all inputs
        x = cat(x, 1)

        return self._layers(x).squeeze()

    def loss(self, pred: tensor, y: tensor) -> tensor:
        """
        Calls the criterion and add elastic penalty

        Args:
            pred: prediction of the models
            y: targets

        Returns: tensor with loss value
        """
        # Computations of penalties
        flatten_params = [w.view(-1, 1) for w in self.parameters()]
        l1_penalty = mean(tensor([l1_loss(w, zeros_like(w)) for w in flatten_params]))
        l2_penalty = mean(tensor([mse_loss(w, zeros_like(w)) for w in flatten_params]))

        return self._criterion(pred, y) + self._alpha*l1_penalty + self._beta*l2_penalty

    @abstractmethod
    def predict(self, x_cont: Optional[tensor], x_cat: Optional[tensor], **kwargs) -> tensor:
        """
        Returns the outputs of the model

        Args:
            x_cont: tensor with continuous inputs
            x_cat: tensor with categorical ordinal encodings

        Returns: tensor with regression values or probabilities (not normalized probabilities)
        """
        raise NotImplementedError


class NNRegression(NNModel):
    """
    Neural Network dedicated to regression
    """
    def __init__(self, layers: List[int], activation: str, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, num_cont_col: Optional[int] = None, cat_sizes: Optional[List[int]] = None,
                 **kwargs):

        """
        Builds layers and set criterion

        Args:
            layers: list with number of units in each hidden layer
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns
             (equal to number of class in the case of classification)
            cat_sizes: list of integer representing the size of each categorical column
            kwargs: argument used to avoid problem in NNModelGenerator class
        """

        # We call parent's constructor
        super().__init__(output_size=1, layers=layers, activation=activation, alpha=alpha, beta=beta,
                         criterion=MSELoss(), dropout=dropout, num_cont_col=num_cont_col, cat_sizes=cat_sizes)

    def predict(self, x_cont: Optional[tensor], x_cat: Optional[tensor], **kwargs):
        """
        Returns the real-valued predictions

        Args:
            x_cont: tensor with continuous inputs
            x_cat: tensor with categorical ordinal encodings

        Returns: tensor with regression values
        """

        # We turn in eval mode
        self.eval()

        # We execute a forward pass
        with no_grad():
            output = self(x_cont, x_cat)

        return output


class NNClassifier(NNModel):
    """
    Neural Network dedicated to classification
    """
    def __init__(self, output_size: int, layers: List[int], activation: str, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, num_cont_col: Optional[int] = None,
                 cat_sizes: Optional[List[int]] = None):

        """
        Builds the layers of the model and sets the criterion

        Args:
            output_size: the number of nodes in the last layer of the neural network

            layers: list with number of units in each hidden layer
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns
            (equal to number of class in the case of classification)
            cat_sizes: list of integer representing the size of each categorical column
        """
        # We call parent's constructor
        super().__init__(output_size=output_size, layers=layers, activation=activation, criterion=CrossEntropyLoss(),
                         alpha=alpha, beta=beta, dropout=dropout, num_cont_col=num_cont_col, cat_sizes=cat_sizes)

    def predict(self, x_cont: Optional[tensor], x_cat: Optional[tensor], **kwargs) -> tensor:
        """
        Returns the outputs of the model

        Args:
            x_cont: tensor with continuous inputs
            x_cat: tensor with categorical ordinal encodings

        Returns: tensor with regression values or probabilities (not normalized probabilities)
        """

        # We turn in eval mode
        self.eval()

        # We execute a forward pass and get the log probabilities if needed
        with no_grad():
            output = self(x_cont, x_cat)
            if kwargs.get("log_prob", False):
                log_soft = log_softmax(output, dim=1).float()
                return log_soft
            else:
                return argmax(output, dim=1).long()


