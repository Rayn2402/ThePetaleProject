"""
Filename: han_base_models.py

Author: Nicolas Raymond

Description: Defines all components related to Heterogeneous Graph Attention Network (HAN).
             The code was mainly taken from this DGL code example :
             https://github.com/dmlc/dgl/tree/master/examples/pytorch/han

Date of last modification: 2021/10/26

"""

from dgl import DGLHeteroGraph
from src.data.processing.datasets import MaskType, PetaleStaticGNNDataset
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.models.blocks.gnn_blocks import HANLayer
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import BinaryClassificationMetric, BinaryCrossEntropy, Metric, \
    RegressionMetric, RootMeanSquaredError
from torch import cat, no_grad, ones, sigmoid, tensor
from torch.nn import BCEWithLogitsLoss, Linear, MSELoss
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Tuple


class HAN(TorchCustomModel):
    """
    Heterogeneous Graph Attention Network model.
    """
    def __init__(self,
                 meta_paths: List[List[str]],
                 hidden_size: int,
                 out_size: int,
                 num_heads: int,
                 dropout: float,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 cat_idx: List[int],
                 cat_sizes: List[int],
                 cat_emb_sizes: List[int],
                 num_cont_col: Optional[int] = None,
                 alpha: float = 0,
                 beta: float = 0,
                 verbose: bool = False
                 ):
        """
        Creates n HAN layers, where n is the number of attention heads

        Args:
            meta_paths: list of metapaths, each meta path is a list of edge types
            hidden_size: size of embedding learnt within each attention head
            out_size: output size (number of node in last layer)
            num_heads: number of attention heads in the HANLayer
            dropout: dropout probability
            criterion: loss function of our model
            criterion_name: name of the loss function of our model
            eval_metric: evaluation metric
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            verbose: True if we want trace of the training progress
        """
        # Call of parent's constructor
        super().__init__(criterion=criterion,
                         criterion_name=criterion_name,
                         eval_metric=eval_metric,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         verbose=verbose)

        # Initialization of the main layer
        self._gnn_layer = HANLayer(meta_paths=meta_paths,
                                   in_size=self._input_size,
                                   out_size=hidden_size,
                                   layer_num_heads=num_heads,
                                   dropout=dropout)

        # Addition of linear layer before calculation of the loss
        self._linear_layer = Linear(hidden_size * num_heads, out_size)

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

        # We extract the features related to all the train mask
        x, _, _ = dataset[train_mask]

        # We execute one training step
        for item in train_loader:

            # We extract the data
            _, y, idx = item

            # We map original idx to their position in the train mask
            pos_idx = [train_idx_map[i.item()] for i in idx]

            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the forward pass
            output = self(train_subgraph, x)

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
        self._evaluations[MaskType.TRAIN][self._criterion_name].append(mean_epoch_loss)
        self._evaluations[MaskType.TRAIN][self._eval_metric.name].append(epoch_score / nb_batch)

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

        # We check if there is validation to do
        if valid_loader is None:
            return False

        # We extract valid_subgraph, mask (train + valid) and valid_idx_map
        valid_subgraph, valid_idx_map, mask = dataset.valid_subgraph

        # We extract the features related to all the train + valid
        x, _, _ = dataset[mask]

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
                output = self(valid_subgraph, x)

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
        self._evaluations[MaskType.VALID][self._criterion_name].append(mean_epoch_loss)
        self._evaluations[MaskType.VALID][self._eval_metric.name].append(mean_epoch_score)

        # We check early stopping status
        early_stopper(mean_epoch_score, self)

        if early_stopper.early_stop:
            return True

        return False

    def forward(self,
                g: DGLHeteroGraph,
                x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            g: DGL Heterogeneous graph
            x: (N,D) tensor with D-dimensional samples

        Returns: (N, D') tensor with values of the node within the last layer

        """

        # We perform entity embeddings and concatenate all inputs
        if len(self._cont_idx) != 0:
            x = cat([x[:, self._cont_idx], self._embedding_block(x)], 1)
        else:
            x = self._embedding_block(x)

        # We make a forward pass through the han main layer to get the embeddings
        h = self._gnn_layer(g, x)

        # We pass the final embedding through a linear layer
        return self._linear_layer(h).squeeze()


class HANBinaryClassifier(HAN):
    """
    Single layered heterogeneous graph attention network binary classifier
    """
    def __init__(self,
                 meta_paths: List[List[str]],
                 hidden_size: int,
                 num_heads: int,
                 dropout: float,
                 cat_idx: List[int],
                 cat_sizes: List[int],
                 cat_emb_sizes: List[int],
                 eval_metric: Optional[BinaryClassificationMetric] = None,
                 num_cont_col: Optional[int] = None,
                 alpha: float = 0,
                 beta: float = 0,
                 verbose: bool = False
                 ):
        """
        Sets protected attributes of the HAN model

        Args:
            meta_paths: list of metapaths, each meta path is a list of edge types
            hidden_size: size of embedding learnt within each attention head
            num_heads: int representing the number of attention heads
            dropout: dropout probability
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            verbose: true to print training progress when fit is called
        """
        # Call parent's constructor
        eval_metric = eval_metric if eval_metric is not None else BinaryCrossEntropy()
        super().__init__(meta_paths=meta_paths,
                         hidden_size=hidden_size,
                         out_size=1,
                         num_heads=num_heads,
                         dropout=dropout,
                         criterion=BCEWithLogitsLoss(reduction='none'),
                         criterion_name='WBCE',
                         eval_metric=eval_metric,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
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
            x, _, _ = dataset[mask_with_train]
            return sigmoid(self(g, x))[pos_idx]


class HANRegressor(HAN):
    """
    Single layered heterogeneous graph attention network regression model
    """
    def __init__(self,
                 meta_paths: List[List[str]],
                 hidden_size: int,
                 num_heads: int,
                 dropout: float,
                 cat_idx: List[int],
                 cat_sizes: List[int],
                 cat_emb_sizes: List[int],
                 eval_metric: Optional[RegressionMetric] = None,
                 num_cont_col: Optional[int] = None,
                 alpha: float = 0,
                 beta: float = 0,

                 verbose: bool = False
                 ):
        """
        Sets protected attributes of the HAN model

        Args:
            meta_paths: list of metapaths, each meta path is a list of edge types
            hidden_size: size of embedding learnt within each attention head
            num_heads: int representing the number of attention heads
            dropout: dropout probability
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            verbose: true to print training progress when fit is called
        """
        # Call parent's constructor
        eval_metric = eval_metric if eval_metric is not None else RootMeanSquaredError()
        super().__init__(meta_paths=meta_paths,
                         hidden_size=hidden_size,
                         out_size=1,
                         num_heads=num_heads,
                         dropout=dropout,
                         criterion=MSELoss(reduction='none'),
                         criterion_name='MSE',
                         eval_metric=eval_metric,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
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
            x, _, _ = dataset[mask_with_train]
            return self(g, x)[pos_idx]