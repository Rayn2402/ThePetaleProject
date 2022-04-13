"""
Filename: gcn_base_models.py

Author: Nicolas Raymond

Description: This file defines the Graph Convolutional Network model

Date of last modification: 2022/04/12
"""
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
from src.data.processing.gnn_datasets import PetaleKGNNDataset
from src.models.abstract_models.gnn_base_models import GNN
from src.utils.score_metrics import BinaryCrossEntropy, Metric, RootMeanSquaredError
from torch import cat, no_grad, sigmoid, tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
from typing import Callable, List, Optional


class GCN(GNN):
    """
    Graph Attention Network model
    """
    def __init__(self,
                 output_size: int,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 alpha: float = 0,
                 beta: float = 0,
                 hidden_size: Optional[int] = None,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):

        """
        Builds the layers of the model and sets other protected attributes

        Args:
            criterion: loss function of our model
            criterion_name: name of the loss function
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            hidden_size: size of the hidden states after the graph convolution
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            verbose: True if we want trace of the training progress
        """
        # We call parent's constructor
        super().__init__(hidden_size=hidden_size,
                         criterion=criterion,
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
        self._conv_layer = GraphConv(in_feats=self._input_size,
                                     out_feats=self._hidden_size,
                                     norm='none',
                                     allow_zero_in_degree=False)

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
        x = cat(new_x, 1)

        # We apply the graph convolutional layer
        h = self._conv_layer(g, x, edge_weight=g.edata['w'])

        # We apply the residual connection
        h = self._dropout(self._bn(cat([h, x], dim=1)))

        # We apply the linear layer
        return self._linear_layer(h).squeeze()

class GCNClassifier(GCN):
    """
    Graph Convolutional Network classification model
    """
    def __init__(self,
                 eval_metric: Metric,
                 alpha: float = 0,
                 beta: float = 0,
                 hidden_size: Optional[int] = None,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 pos_weight: Optional[float] = None,
                 verbose: bool = False):
        """
        Sets the eval metric and the other attributes using the parent constructor

        Args:
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            hidden_size: size of the hidden states after the graph convolution
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            pos_weight: scaling factor attributed to positive samples (samples in class 1)
            verbose: True if we want trace of the training progress
        """
        # We set the eval metric
        if eval_metric is None:
            eval_metric = BinaryCrossEntropy(pos_weight=pos_weight)
        else:
            if hasattr(eval_metric, 'pos_weight'):
                eval_metric.pos_weight = pos_weight

        super().__init__(output_size=1,
                         hidden_size=hidden_size,
                         criterion=BCEWithLogitsLoss(pos_weight=pos_weight),
                         criterion_name='BCE',
                         eval_metric=eval_metric,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         verbose=verbose)

    def predict_proba(self,
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
            return sigmoid(self(g, x)[pos_idx])


class GCNRegressor(GCN):
    """
    Graph Convolutional Network regression model
    """
    def __init__(self,
                 eval_metric: Metric,
                 alpha: float = 0,
                 beta: float = 0,
                 hidden_size: Optional[int] = None,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):
        """
        Sets the attributes using the parent constructor

        Args:
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            hidden_size: size of the hidden states after the graph convolution
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
                         criterion=MSELoss(),
                         criterion_name='MSE',
                         eval_metric=eval_metric,
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
