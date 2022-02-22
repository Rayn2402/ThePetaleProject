"""
Filename: gat_base_models.py

Author: Nicolas Raymond

Description: This file defines the Graph Attention Network model

Date of last modification: 2022/02/22
"""
from dgl import DGLGraph
from dgl.nn.pytorch import GATv2Conv
from src.data.processing.gnn_datasets import PetaleKGNNDataset
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import Metric
from torch import cat, tensor
from torch.nn import Linear
from torch.nn.functional import elu
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Union, Tuple


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
                 dropout: float = 0,
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
            dropout: probability of dropout
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
        self._conv_layer = GATv2Conv(in_feats=self._input_size,
                                     out_feats=hidden_size,
                                     num_heads=num_heads,
                                     feat_drop=dropout,
                                     attn_drop=dropout,
                                     activation=elu)

        # We build the final linear layer
        self._linear_layer = Linear(hidden_size*num_heads, output_size)

    def _execute_train_step(self, train_data: Tuple[DataLoader, PetaleKGNNDataset],
                            sample_weights: tensor) -> float:
        raise NotImplementedError

    def _execute_valid_step(self, valid_data: Optional[Union[DataLoader, Tuple[DataLoader, PetaleKGNNDataset]]],
                            early_stopper: Optional[EarlyStopper]) -> bool:
        raise NotImplementedError

    def forward(self,
                g: DGLGraph,
                x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            g: Homogeneous bidirected population graph
            x: (N,D) tensor with D-dimensional samples

        Returns: (N, D') tensor with values of the node within the last layer
        """
        # We initialize a list of tensors to concatenate
        new_x = []

        # We extract continuous data
        if len(self._cont_idx) != 0:
            new_x.append(x[:, self._cont_idx])

        # We perform entity embeddings on categorical features not identified as genes
        if len(self._cat_idx) != 0:
            new_x.append(self._embedding_block(x))

        # We concatenate all inputs
        h = cat(new_x, 1)

        # We apply the graph convolutional layer
        h = self._conv_layer(g, h)

        # We apply the linear layer
        return self._linear_layer(h).squeeze()
