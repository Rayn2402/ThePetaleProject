"""
Filename: han_base_models.py

Author: Nicolas Raymond

Description: Defines all components related to Heterogeneous Graph Attention Network (HAN).
             The code was mainly taken from this DGL code example :
             https://github.com/dmlc/dgl/tree/master/examples/pytorch/han

Date of last modification: 2021/10/26

"""

from dgl import DGLHeteroGraph
from src.data.processing.datasets import PetaleStaticGNNDataset
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.models.blocks.gnn_blocks import HANLayer
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import BinaryClassificationMetric, BinaryCrossEntropy, Metric, \
    RegressionMetric, RootMeanSquaredError
from torch import no_grad, ones, sigmoid, tensor
from torch.nn import BCEWithLogitsLoss, Linear, ModuleList, MSELoss
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Tuple


class HAN(TorchCustomModel):
    """
    Heterogeneous Graph Attention Network model.
    """
    def __init__(self,
                 meta_paths: List[List[str]],
                 in_size: int,
                 hidden_size: int,
                 out_size: int,
                 num_heads: List[int],
                 dropout: float,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 alpha: float = 0,
                 beta: float = 0,
                 verbose: bool = False
                 ):
        """
        Creates n HAN layers, where n is the number of attention heads

        Args:
            meta_paths: list of metapaths, each meta path is a list of edge types
            in_size: input size (number of features per node)
            hidden_size: size of embedding learnt within each attention head
            out_size: output size (number of node in last layer)
            num_heads: list with int representing the number of attention heads per layer
            dropout: dropout probability
            criterion: loss function of our model
            criterion_name: name of the loss function of our model
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
        """
        # Call of parent's constructor
        super().__init__(criterion=criterion,
                         criterion_name=criterion_name,
                         eval_metric=eval_metric,
                         alpha=alpha,
                         beta=beta,
                         verbose=verbose)

        # Initialization of layers (nb of layers = length of num heads list)
        self.gnn_layers = ModuleList()
        self.gnn_layers.append(HANLayer(meta_paths=meta_paths,
                                        in_size=in_size,
                                        out_size=hidden_size,
                                        layer_num_heads=num_heads[0],
                                        dropout=dropout))
        for l in range(1, len(num_heads)):
            self.gnn_layers.append(HANLayer(meta_paths=meta_paths,
                                            in_size=hidden_size * num_heads[l-1],
                                            out_size=hidden_size,
                                            layer_num_heads=num_heads[l],
                                            dropout=dropout))

        # Addition of linear layer before calculation of the loss
        self.linear_layer = Linear(hidden_size * num_heads[-1], out_size)

        # Attribute dedicated to training
        self._optimizer = None

    def _execute_train_step(self,
                            train_data: Tuple[DataLoader, PetaleStaticGNNDataset],
                            sample_weights: tensor) -> float:
        """
        Executes one training epoch

        Args:
            train_data: tuple (train loader, dataset)
            sample_weights: weights of the samples in the loss

        Returns: mean epoch loss
        """

        # We set the model for training
        self.train()
        epoch_loss, epoch_score = 0, 0

        # We extract train loader, dataset
        train_loader, dataset = train_data

        # We extract train_subgraph, train_mask and train_idx_map
        train_subgraph, train_idx_map, train_mask = dataset.train_subgraph

        # We execute one training step
        for item in train_loader:

            # We extract the data
            _, y, idx = item

            # We map original idx to their position in the train mask
            pos_idx = [train_idx_map[i.item()] for i in idx]

            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the forward pass
            output = self(train_subgraph, dataset.x_cont[train_mask])

            # We calculate the loss and the score
            loss = self.loss(sample_weights[idx], output[pos_idx], y)
            score = self._eval_metric(output[pos_idx], y)
            epoch_loss += loss.item()
            epoch_score += score

            # We perform the backward pass
            loss.backward()

            # We perform a single optimization step (parameter update)
            self._optimizer.step()

        # We save mean epoch loss and mean epoch score
        nb_batch = len(train_data)
        mean_epoch_loss = epoch_loss / nb_batch
        self._evaluations["train"][self._criterion_name].append(mean_epoch_loss)
        self._evaluations["train"][self._eval_metric.name].append(epoch_score / nb_batch)

        return mean_epoch_loss

    def _execute_valid_step(self,
                            valid_data: Optional[Tuple[DataLoader, PetaleStaticGNNDataset]],
                            early_stopper: EarlyStopper) -> bool:
        """
        Executes an inference step on the validation data and apply early stopping if needed

        Args:
            valid_data: tuple (valid loader, dataset)
            early_stopper: early stopper keeping track of validation loss

        Returns: True if we need to early stop
        """
        # We extract train loader, dataset
        valid_loader, dataset = valid_data

        # We extract valid_subgraph, mask (train + valid) and valid_idx_map
        valid_subgraph, valid_idx_map, mask = dataset.valid_subgraph

        # We check if there is validation to do
        if valid_loader is None:
            return False

        # Set model for evaluation
        self.eval()
        epoch_loss, epoch_score = 0, 0

        # We execute one inference step on validation set
        with no_grad():

            for item in valid_loader:

                # We extract the data
                _, y, idx = item

                # We map original idx to their position in the train mask
                pos_idx = [valid_idx_map[i.item()] for i in idx]

                # We perform the forward pass: compute predicted outputs by passing inputs to the model
                output = self(valid_subgraph, dataset.x_cont[mask])

                # We calculate the loss and the score
                batch_size = len(idx)
                sample_weights = ones(batch_size) / batch_size  # Sample weights are equal for validation (1/N)
                loss = self.loss(sample_weights, output[pos_idx], y)
                score = self._eval_metric(output[pos_idx], y)
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

    def forward(self,
                g: DGLHeteroGraph,
                h: tensor) -> tensor:

        # We make a forward pass through han layers
        for gnn in self.gnn_layers:
            h = gnn(g, h)

        # We pass the final embedding through a linear layer
        return self.linear_layer(h).squeeze()


class HANBinaryClassifier(HAN):
    """
    Single layered heterogeneous graph attention network binary classifier
    """
    def __init__(self,
                 meta_paths: List[List[str]],
                 in_size: int,
                 hidden_size: int,
                 num_heads: int,
                 dropout: float,
                 eval_metric: Optional[BinaryClassificationMetric] = None,
                 alpha: float = 0,
                 beta: float = 0,
                 verbose: bool = False
                 ):
        """
        Sets protected attributes of the HAN model

        Args:
            meta_paths: list of metapaths, each meta path is a list of edge types
            in_size: input size (number of features per node)
            hidden_size: size of embedding learnt within each attention head
            num_heads: int representing the number of attention heads
            dropout: dropout probability
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            verbose: true to print training progress when fit is called
        """
        # Call parent's constructor
        eval_metric = eval_metric if eval_metric is not None else BinaryCrossEntropy()
        super().__init__(meta_paths=meta_paths,
                         in_size=in_size,
                         hidden_size=hidden_size,
                         out_size=1,
                         num_heads=[num_heads],
                         dropout=dropout,
                         criterion=BCEWithLogitsLoss(reduction='none'),
                         criterion_name='WBCE',
                         eval_metric=eval_metric,
                         alpha=alpha,
                         beta=beta,
                         verbose=verbose)

    def predict_proba(self,
                      dataset: PetaleStaticGNNDataset,
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
        # We extract subgraph data (we add training data for graph convolution)
        if mask is not None:
            mask_with_train = list(set(mask + dataset.train_mask))
            g, idx_map = dataset.get_arbitrary_subgraph(mask_with_train)
        else:
            mask = dataset.test_mask
            g, idx_map, mask_with_train = dataset.test_subgraph

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a softmax
        with no_grad():
            pos_idx = [idx_map[i] for i in mask]
            return sigmoid(self(g, dataset.x_cont[mask_with_train]))[pos_idx]


class HANRegressor(HAN):
    """
    Single layered heterogeneous graph attention network regression model
    """
    def __init__(self,
                 meta_paths: List[List[str]],
                 in_size: int,
                 hidden_size: int,
                 num_heads: int,
                 dropout: float,
                 eval_metric: Optional[RegressionMetric] = None,
                 alpha: float = 0,
                 beta: float = 0,
                 verbose: bool = False
                 ):
        """
        Sets protected attributes of the HAN model

        Args:
            meta_paths: list of metapaths, each meta path is a list of edge types
            in_size: input size (number of features per node)
            hidden_size: size of embedding learnt within each attention head
            num_heads: int representing the number of attention heads
            dropout: dropout probability
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            verbose: true to print training progress when fit is called
        """
        # Call parent's constructor
        eval_metric = eval_metric if eval_metric is not None else RootMeanSquaredError()
        super().__init__(meta_paths=meta_paths,
                         in_size=in_size,
                         hidden_size=hidden_size,
                         out_size=1,
                         num_heads=[num_heads],
                         dropout=dropout,
                         criterion=MSELoss(reduction='none'),
                         criterion_name='MSE',
                         eval_metric=eval_metric,
                         alpha=alpha,
                         beta=beta,
                         verbose=verbose)

    def predict(self,
                dataset: PetaleStaticGNNDataset,
                mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the real-valued predictions for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: list of dataset idx for which we want to predict target

        Returns: (N,) tensor
        """
        # We extract subgraph data (we add training data for graph convolution)
        if mask is not None:
            mask_with_train = list(set(mask + dataset.train_mask))
            g, idx_map = dataset.get_arbitrary_subgraph(mask_with_train)
        else:
            mask = dataset.test_mask
            g, idx_map, mask_with_train = dataset.test_subgraph

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a softmax
        with no_grad():
            pos_idx = [idx_map[i] for i in mask]
            return self(g, dataset.x_cont[mask_with_train])[pos_idx]