"""
Filename: mlp_base_models.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to define the MLP model with entity embeddings
             and its children MLPBinaryClassifier and PetaleMLPRegressor.
             These models are not shaped to inherit from the PetaleRegressor
             and PetaleBinaryClassifier classes. However, two wrapper classes for torch models
             are provided to enable the use of these mlp models with hyperparameter tuning functions.

Date of last modification : 2021/11/09
"""

from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.models.blocks.mlp_blocks import BaseBlock
from src.data.processing.datasets import MaskType, PetaleDataset
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import BinaryCrossEntropy, Metric, RootMeanSquaredError
from torch import cat, no_grad, tensor, ones, sigmoid
from torch.nn import BCEWithLogitsLoss, Linear, MSELoss, Sequential
from torch.utils.data import DataLoader
from typing import Callable, List, Optional


class MLP(TorchCustomModel):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self,
                 output_size: int,
                 layers: List[int],
                 activation: str,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 sam: bool = True,
                 verbose: bool = False):

        """
        Builds the layers of the model and sets other protected attributes

        Args:
            output_size: the number of nodes in the last layer of the neural network
            layers: list with number of units in each hidden layer
            criterion: loss function of our model
            criterion_name: name of the loss function
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            sam: true to use Sharpness-Aware Minimization (SAM)
            verbose: True if we want trace of the training progress
        """
        if num_cont_col is None and cat_sizes is None:
            raise ValueError("There must be continuous columns or categorical columns")

        # We call parent's constructor
        super().__init__(criterion=criterion,
                         criterion_name=criterion_name,
                         eval_metric=eval_metric,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         sam=sam,
                         verbose=verbose)

        # We create the main layers of our model
        layers.insert(0, self._input_size)
        all_layers = []
        if len(layers) > 1:
            all_layers = [BaseBlock(input_size=layers[i-1],
                                    output_size=layers[i],
                                    activation=activation,
                                    p=dropout) for i in range(1, len(layers))]

        # We add a linear layer to complete the layers
        self._layers = Sequential(*all_layers, Linear(layers[-1], output_size))

    def _execute_train_step(self,
                            train_data: DataLoader,
                            sample_weights: tensor) -> float:
        """
        Executes one training epoch

        Args:
            train_data: training dataloader
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
        self._evaluations[MaskType.TRAIN][self._criterion_name].append(mean_epoch_loss)
        self._evaluations[MaskType.TRAIN][self._eval_metric.name].append(epoch_score/nb_batch)

        return mean_epoch_loss

    def _execute_valid_step(self,
                            valid_loader: Optional[DataLoader],
                            early_stopper: EarlyStopper) -> bool:
        """
        Executes an inference step on the validation data

        Args:
            valid_loader: validation dataloader
            early_stopper: early stopper keeping track of the validation loss

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
        self._evaluations[MaskType.VALID][self._criterion_name].append(mean_epoch_loss)
        self._evaluations[MaskType.VALID][self._eval_metric.name].append(mean_epoch_score)

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

        Returns: (N, D') tensor with values of the node within the last layer

        """
        # We initialize a list of tensors to concatenate
        new_x = []

        # We extract continuous data
        if len(self._cont_idx) != 0:
            new_x.append(x[:, self._cont_idx])

        # We perform entity embeddings
        if len(self._cat_idx) != 0:
            new_x.append(self._embedding_block(x))

        # We concatenate all inputs
        x = cat(new_x, 1)

        return self._layers(x).squeeze()


class MLPBinaryClassifier(MLP):
    """
    Multilayer perceptron binary classification model with entity embedding
    """
    def __init__(self,
                 layers: List[int],
                 activation: str,
                 eval_metric: Optional[Metric] = None,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 sam: bool = True,
                 verbose: bool = False):
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
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            sam: true to use Sharpness-Aware Minimization (SAM)
            verbose: true to print training progress when fit is called
        """
        eval_metric = eval_metric if eval_metric is not None else BinaryCrossEntropy()
        super().__init__(output_size=1,
                         layers=layers,
                         activation=activation,
                         criterion=BCEWithLogitsLoss(reduction='none'),
                         criterion_name='WBCE',
                         eval_metric=eval_metric,
                         dropout=dropout,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         sam=sam,
                         verbose=verbose)

    def predict_proba(self,
                      dataset: PetaleDataset,
                      mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the probabilities of being in class 1 for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: list of dataset idx for which we want to predict proba

        Returns: (N,) tensor
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a sigmoid
        with no_grad():
            return sigmoid(self(x))


class MLPRegressor(MLP):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self,
                 layers: List[int],
                 activation: str,
                 eval_metric: Optional[Metric] = None,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 sam: bool = True,
                 verbose: bool = False):
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
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            sam: true to use Sharpness-Aware Minimization (SAM)
            verbose: true to print training progress when fit is called
        """
        eval_metric = eval_metric if eval_metric is not None else RootMeanSquaredError()
        super().__init__(output_size=1,
                         layers=layers,
                         activation=activation,
                         criterion=MSELoss(reduction='none'),
                         criterion_name='MSE',
                         eval_metric=eval_metric,
                         dropout=dropout,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         sam=sam,
                         verbose=verbose)

    def predict(self,
                dataset: PetaleDataset,
                mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the predicted real-valued targets for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset
            mask: list of dataset idx for which we want to make predictions

        Returns: (N,) tensor
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # Extraction of data
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass
        with no_grad():
            return self(x)
