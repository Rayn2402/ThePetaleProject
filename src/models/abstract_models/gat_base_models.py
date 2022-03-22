"""
Filename: gat_base_models.py

Author: Nicolas Raymond

Description: This file defines the Graph Attention Network model

Date of last modification: 2022/02/22
"""
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv
from src.data.processing.gnn_datasets import MaskType, PetaleKGNNDataset
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import Metric, RootMeanSquaredError
from torch import cat, no_grad, ones, tensor
from torch.nn import BatchNorm1d, Identity, Linear, MSELoss
from torch.nn.functional import elu, relu
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Union, Tuple


class GAT(TorchCustomModel):
    """
    Graph Attention Network model
    """
    def __init__(self,
                 output_size: int,
                 hidden_size: int,
                 num_heads: int,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 feat_dropout: float = 0,
                 attn_dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):

        """
        Builds the layers of the model and sets other protected attributes

        Args:
            hidden_size: size of the hidden states after the graph convolution
            num_heads: number of attention heads
            criterion: loss function of our model
            criterion_name: name of the loss function
            eval_metric: evaluation metric
            feat_dropout: features dropout probability
            attn_dropout: attention dropout probability
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            verbose: True if we want trace of the training progress
        """
        # We call parent's constructor
        super().__init__(criterion=criterion,
                         criterion_name=criterion_name,
                         eval_metric=eval_metric,
                         output_size=output_size,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         verbose=verbose)

        # We build the main layer
        self._conv_layer = GATConv(in_feats=self._input_size,
                                   out_feats=hidden_size,
                                   num_heads=num_heads,
                                   feat_drop=feat_dropout,
                                   attn_drop=attn_dropout,
                                   activation=relu)

        # We save the batch norm layer
        self._bn = BatchNorm1d(hidden_size)

        # We save the number of attention heads
        self._num_att_heads = num_heads

        # We save the linear layer
        if hidden_size != output_size:
            self._linear_layer = Linear(hidden_size, output_size)
        else:
            self._linear_layer = Identity()

    def _execute_train_step(self, train_data: Tuple[DataLoader, PetaleKGNNDataset],
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

        # We extract the training dataloader and the complete dataset
        train_loader, dataset = train_data

        # We extract train_subgraph, train_mask and train_idx_map
        train_subgraph, train_idx_map, train_mask = dataset.train_subgraph

        # We extract the features related to all the train mask
        x, _, _ = dataset[train_mask]

        # We execute one training step
        for item in train_loader:

            # We extract the data
            _, y, idx = item

            # We map the original idx to their position in the train mask
            pos_idx = [train_idx_map[i.item()] for i in idx]

            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the weight update
            pred, loss = self._update_weights(sample_weights[idx], [train_subgraph, x], y, pos_idx)

            # We update the metrics history
            score = self._eval_metric(pred, y)
            epoch_loss += loss
            epoch_score += score

        # We save mean epoch loss and mean epoch score
        nb_batch = len(train_data)
        mean_epoch_loss = epoch_loss / nb_batch
        self._evaluations[MaskType.TRAIN][self._criterion_name].append(mean_epoch_loss)
        self._evaluations[MaskType.TRAIN][self._eval_metric.name].append(epoch_score / nb_batch)

        return mean_epoch_loss

    def _execute_valid_step(self, valid_data: Optional[Union[DataLoader, Tuple[DataLoader, PetaleKGNNDataset]]],
                            early_stopper: Optional[EarlyStopper]) -> bool:
        """
        Executes an inference step on the validation data and apply early stopping if needed

        Args:
            valid_data: tuple (valid loader, dataset)
            early_stopper: early stopper keeping track of validation loss

        Returns: True if we need to early stop
        """
        # We check if there is validation to do
        if valid_data is None:
            return False

        # We extract train loader, dataset
        valid_loader, dataset = valid_data

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

                # We perform the forward pass
                pred = self(valid_subgraph, x)

                # We calculate the loss and the score
                batch_size = len(idx)
                sample_weights = ones(batch_size) / batch_size  # Sample weights are equal for validation (1/N)
                epoch_loss += self.loss(sample_weights, pred[pos_idx], y).item()
                epoch_score += self._eval_metric(pred[pos_idx], y)

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
                g: DGLGraph,
                x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            g: Homogeneous directed population graph
            x: (N,D) tensor with D-dimensional samples

        Returns: (N, D') tensor with values of the node within the last layer
        """
        # We initialize a list of tensors to concatenate
        new_x = []

        # We extract continuous data
        if len(self._cont_idx) != 0:
            new_x.append(x[:, self._cont_idx])

        # We perform entity embeddings on categorical features
        if len(self._cat_idx) != 0:
            new_x.append(self._embedding_block(x))

        # We concatenate all inputs
        h = cat(new_x, 1)

        # We apply the graph convolutional layer
        h = self._conv_layer(g, h)

        # We take the average of all the attention heads and apply batch norm
        h = self._bn(h.sum(dim=1)/self._num_att_heads)

        # We apply the linear layer
        return self._linear_layer(h).squeeze()


class GATRegressor(GAT):
    """
    Graph Attention Network regression model
    """
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 eval_metric: Metric,
                 feat_dropout: float = 0,
                 attn_dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):
        """
        Sets the attributes using the parent constructor

        Args:
            hidden_size: size of the hidden states after the graph convolution
            num_heads: number of attention heads
            eval_metric: evaluation metric
            feat_dropout: features dropout probability
            attn_dropout: attention dropout probability
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            verbose: True if we want trace of the training progress
        """
        # We call parent's constructor
        eval_metric = eval_metric if eval_metric is not None else RootMeanSquaredError()
        super().__init__(output_size=1,
                         hidden_size=hidden_size,
                         num_heads=num_heads,
                         criterion=MSELoss(reduction='none'),
                         criterion_name='MSE',
                         eval_metric=eval_metric,
                         feat_dropout=feat_dropout,
                         attn_dropout=attn_dropout,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         verbose=verbose)

    def predict(self,
                dataset: PetaleKGNNDataset,
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
        if mask is None or all([i in dataset.test_mask for i in mask]):
            mask = dataset.test_mask
            g, idx_map, mask_with_remaining_idx = dataset.test_subgraph

        # We extract subgraph data (we add training data for graph convolution)
        else:
            g, idx_map, mask_with_remaining_idx = dataset.get_arbitrary_subgraph(mask)

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a softmax
        with no_grad():
            pos_idx = [idx_map[i] for i in mask]
            x, _, _ = dataset[mask_with_remaining_idx]
            return self(g, x)[pos_idx]
