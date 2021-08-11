"""
Authors: Nicolas Raymond
         Mehdi Mitiche

This file is used to store the MLP with entity embeddings base model and its children PetaleMLPClassifier
and PetaleMLPRegressor
"""

from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.data.processing.datasets import PetaleDataset
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import Metric, BinaryCrossEntropy, RootMeanSquaredError
from torch import cat, nn, no_grad, tensor, ones, sigmoid
from torch.nn import ModuleList, Embedding,\
    Linear, BatchNorm1d, Dropout, Sequential, BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from typing import Callable, List, Optional


class MLP(TorchCustomModel):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self, output_size: int, layers: List[int], activation: str,
                 criterion: Callable, criterion_name: str, eval_metric: Metric, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None, cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None, verbose: bool = False):

        """
        Builds the layers of the model and sets other protected attributes

        Args:
            output_size: the number of nodes in the last layer of the neural network
            layers: list with number of units in each hidden layer
            criterion: loss function of our model
            criterion_name: name of the loss function of our model
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns
                          (equal to number of class in the case of classification)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            verbose: True if we want trace of the training progress
        """
        assert num_cont_col is not None or cat_sizes is not None, "There must be continuous columns" \
                                                                  " or categorical columns"
        # We call parent's constructor
        super().__init__(criterion=criterion, criterion_name=criterion_name, eval_metric=eval_metric,
                         alpha=alpha, beta=beta, verbose=verbose)

        # We set protected attributes proper to the model
        self._cat_idx = cat_idx if cat_idx is not None else []
        self._cont_idx = [i for i in range(len(self._cat_idx) + num_cont_col) if i not in self._cat_idx]
        self._embedding_layers = None

        # We initialize the input_size
        input_size = num_cont_col if num_cont_col is not None else 0

        # We set the embedding layers
        if cat_idx is not None and cat_sizes is not None:

            # We check embedding sizes (if nothing provided -> emb_sizes = cat_sizes)
            cat_emb_sizes = cat_emb_sizes if cat_emb_sizes is not None else cat_sizes

            # We generate the embedding sizes
            embedding_sizes = [(cat_size, emb_size) for cat_size, emb_size in zip(cat_sizes, cat_emb_sizes)]

            # We create the embedding layers
            self._embedding_layers = ModuleList([Embedding(num_embedding, embedding_dim) for
                                                 num_embedding, embedding_dim in embedding_sizes])
            # We sum the length of all embeddings
            input_size += sum(cat_sizes)

        # We create the different layers of our model
        # Linear --> Activation --> Batch Normalization --> Dropout
        all_layers = []
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

    def _execute_train_step(self, train_data: DataLoader, sample_weights: tensor) -> float:
        """
        Executes one training epoch

        Args:
            train_data: training data loader
            sample_weights: weights of the samples in the loss

        Returns: mean epoch loss
        """
        # We set the model for training
        self.train()
        epoch_loss, epoch_score = 0, 0

        # We execute one training step
        for item in train_data:

            # We extract the data
            x, y, idx = item

            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the forward pass
            output = self(x)

            # We calculate the loss and the score
            loss = self.loss(sample_weights[idx], output, y)
            score = self._eval_metric(output, y)
            epoch_loss += loss.item()
            epoch_score += score

            # We perform the backward pass
            loss.backward()

            # We perform a single optimization step (parameter update)
            self._optimizer.step()

        # We save mean epoch loss and mean epoch score
        nb_batch = len(train_data)
        mean_epoch_loss = epoch_loss/nb_batch
        self._evaluations["train"][self._criterion_name].append(mean_epoch_loss)
        self._evaluations["train"][self._eval_metric.name].append(epoch_score/nb_batch)

        return mean_epoch_loss

    def _execute_valid_step(self, valid_loader: Optional[DataLoader], early_stopper: EarlyStopper) -> bool:
        """
        Executes an inference step on the validation data

        Args:
            valid_loader: validation data loader
            early_stopper: early stopper keeping track of validation loss

        Returns: True if we need to early stop
        """
        if valid_loader is None:
            return False

        # Set model for evaluation
        self.eval()
        epoch_loss, epoch_score = 0, 0

        # We execute one inference step on validation set
        with no_grad():

            for item in valid_loader:

                # We extract the data
                x, y, idx = item

                # We perform the forward pass: compute predicted outputs by passing inputs to the model
                output = self(x)

                # We calculate the loss and the score
                batch_size = len(idx)
                sample_weights = ones(batch_size)/batch_size
                loss = self.loss(sample_weights, output, y)  # Sample weights are equal for validation (1/N)
                score = self._eval_metric(output, y)
                epoch_loss += loss.item()
                epoch_score += score

        # We save mean epoch loss and mean epoch score
        nb_batch = len(valid_loader)
        mean_epoch_loss = epoch_loss / nb_batch
        mean_epoch_score = epoch_score / nb_batch
        self._evaluations["valid"][self._criterion_name].append(mean_epoch_loss)
        self._evaluations["valid"][self._eval_metric.name].append(mean_epoch_score)

        # We check early stopping status
        early_stopper(mean_epoch_score, self)

        if early_stopper.early_stop:
            return True

        return False

    def forward(self, x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: tensor with values of the node from the last layer

        """
        # We initialize list of tensors to concatenate
        new_x = []

        # We extract continuous data
        if len(self._cont_idx) != 0:
            new_x.append(x[:, self._cont_idx])

        # We perform entity embeddings
        if len(self._cat_idx) != 0:
            x_cat = x[:, self._cat_idx]
            embeddings = []
            for i, e in enumerate(self._embedding_layers):
                embeddings.append(e(x_cat[:, i].long()))

            # We concatenate all the embeddings
            new_x.append(cat(embeddings, 1))

        # We concatenate all inputs
        x = cat(new_x, 1)

        return self._layers(x).squeeze()


class MLPBinaryClassifier(MLP):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self, layers: List[int], activation: str,
                 eval_metric: Optional[Metric] = None, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None, cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None, verbose: bool = False):
        """
        Sets protected attributes using parent's constructor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns
                          (equal to number of class in the case of classification)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
        """
        eval_metric = eval_metric if eval_metric is not None else BinaryCrossEntropy()
        super().__init__(output_size=1, layers=layers, activation=activation,
                         criterion=BCEWithLogitsLoss(reduction='none'), criterion_name='WBCE',
                         eval_metric=eval_metric, dropout=dropout, alpha=alpha, beta=beta,
                         num_cont_col=num_cont_col, cat_idx=cat_idx, cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes, verbose=verbose)

    def predict_proba(self, dataset: PetaleDataset, mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the probabilities of being in class 1 for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDatasets which items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to predict proba

        Returns: (N,) tensor or array
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a softmax
        with no_grad():
            return sigmoid(self(x))


class MLPRegressor(MLP):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self, layers: List[int], activation: str,
                 eval_metric: Optional[Metric] = None, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None, cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None, verbose: bool = False):
        """
        Sets protected attributes using parent's constructor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns
                          (equal to number of class in the case of classification)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
        """
        eval_metric = eval_metric if eval_metric is not None else RootMeanSquaredError()
        super().__init__(output_size=1, layers=layers, activation=activation,
                         criterion=MSELoss(reduction='none'), criterion_name='MSE',
                         eval_metric=eval_metric, dropout=dropout, alpha=alpha, beta=beta,
                         num_cont_col=num_cont_col, cat_idx=cat_idx, cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes, verbose=verbose)

    def predict(self, dataset: PetaleDataset) -> tensor:
        """
        Returns the predicted real-valued targets for all samples in the test set

        Args:
            dataset: PetaleDatasets which items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: (N,) array
        """
        # Extraction of test set
        x_test, _, _ = dataset[dataset.test_mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a softmax
        with no_grad():
            return self(x_test)
