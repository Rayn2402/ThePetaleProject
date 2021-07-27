"""
Author: Nicolas Raymond

This file stores all components related to Heterogeneous Graph Attention Network (HAN).
The code was mainly taken from this DGL code example : https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
"""

import dgl

from dgl.nn.pytorch import GATConv
from src.data.processing.datasets import PetaleStaticGNNDataset
from src.models.custom_torch_base import TorchCustomModel
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import Metric
from torch import tensor, softmax, stack, no_grad, ones
from torch.nn import Linear, Module, ModuleList, Sequential, Tanh
from torch.nn.functional import elu
from torch.optim import Adam
from typing import Callable, List, Optional, Union


class SemanticAttention(Module):
    """
    Attention layer that identifies meta paths' importance
    """
    def __init__(self, in_size: int, hidden_size: int = 128):
        """
        Initializes the layers needed to execute the projection from which
        the semantic attention coefficients are calculated

        Args:
            in_size: Input size of embeddings learned with node level attention
            hidden_size: Size of the linear projection used to compute semantic attention
        """
        super(SemanticAttention, self).__init__()

        # Projection = q_t (tanh(Wz + b)) where q is a vector and W is a matrix
        self.project = Sequential(
            Linear(in_size, hidden_size),
            Tanh(),
            Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z: tensor):
        """
        Calculates the final nodes' embeddings combining all embeddings learnt
        within bipartite graphs related to metapaths

        Args:
            z: concatenated embeddings from previous layer

        Returns: final embedding to use with a classifier

        """
        w = self.project(z).mean(0)                     # (M, 1)
        beta = softmax(w, dim=0)                        # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)                        # (N, D * K)


class HANLayer(Module):

    def __init__(self, meta_paths: List[Union[str, List[str]]], in_size: int, out_size: int,
                 layer_num_heads: int, dropout: float):
        """
        Builds a single layer of the Heterogeneous Graph Attention Network model.
        A layers is composed of :
        - One Graph Attention Network for each meta path based graph
        - An additional attention layer to capture meta paths' importance
          and concatenate embedding learnt from every meta paths.

        Args:
            meta_paths: List of metapaths, each meta path is
                        a list of edge types or a string of a single edge type
            in_size: input size (number of features per node)
            out_size: output size (size of the output embedding)
            layer_num_heads: number of attention heads
            dropout: dropout probability
        """

        # Call of module constructor
        super().__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=elu,
                                           allow_zero_in_degree=True))

        # Semantic attention layer
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)

        # Meta paths list
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        # Cached protected attributes
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g: dgl.DGLHeteroGraph, h: tensor) -> tensor:

        # We initialize storage for semantic embeddings
        semantic_embeddings = []

        # We create a list of sub graphs associated to each meta path (if it is not done already)
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)

        # For each meta path we proceed to a forward pass in a GAT
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))

        # We stack the embeddings learnt using each meta path neighborhood
        semantic_embeddings = stack(semantic_embeddings, dim=1)                        # (N, M, D * K)

        # We pass these embeddings through an attention layer
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)


class HAN(TorchCustomModel):
    def __init__(self, meta_paths: List[Union[str, List[str]]], in_size: int, hidden_size: int,
                 out_size: int, num_heads: List[int], dropout: float,
                 criterion: Callable, criterion_name: str, eval_metric: Metric,
                 alpha: float = 0, beta: float = 0, verbose: bool = False
                 ):
        """
        Creates n HAN layers, where n is the number of attention heads

        Args:
            meta_paths: List of metapaths, each meta path is
                        a list of edge types or a string of a single edge type
            in_size: input size (number of features per node)
            hidden_size: size of embedding learnt within each attention head
            out_size: output size (number of node in last layer)
            num_heads: List with int representing the number of attention heads per layer
            dropout: dropout probability
        """
        # Call of parent's constructor
        super().__init__(criterion=criterion, criterion_name=criterion_name, eval_metric=eval_metric,
                         alpha=alpha, beta=beta, verbose=verbose)

        # Initialization of layers (nb of layers = length of num heads list)
        self.gnn_layers = ModuleList()
        self.gnn_layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.gnn_layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                            hidden_size, num_heads[l], dropout))

        # Addition of linear layer before calculation of the loss
        self.linear_layer = Linear(hidden_size * num_heads[-1], out_size)

        # Attribute dedicated to training
        self._optimizer = None

    def _execute_train_step(self, dataset: PetaleStaticGNNDataset, sample_weights: tensor) -> float:
        """
        Executes a single forward pass with all nodes and computes loss
        and gradients using train mask only

        Args:
            dataset: Dataset containing the graph and the nodes' features
            sample_weights: weights of the samples in the loss

        Returns: epoch loss
        """
        # We put the model in train mode
        self.train()

        # We clear the gradients
        self._optimizer.zero_grad()

        # We execute a forward pass and calculate the loss on training set
        train_idx = dataset.train_mask
        output = self(dataset.graph, dataset.x_cont)
        loss = self.loss(sample_weights[train_idx], output[train_idx], dataset.y[train_idx])
        score = self.eval_metric(output[train_idx])

        # We proceed to backpropagation
        loss.backward()
        self._optimizer.step()

        # We save the loss and the score
        self._evaluations["train"][self._criterion_name].append(loss)
        self._evaluations["train"][self._eval_metric.name].append(score)

        return loss

    def _execute_valid_step(self, dataset: PetaleStaticGNNDataset, early_stopper: EarlyStopper) -> bool:
        """
        Executes a validation step to apply early stopping if needed

        Args:
            dataset: Dataset containing the graph and the nodes' features
            early_stopper: object used validate the training progress and prevent overfitting

        Returns: True, if we need to early stop
        """
        # We check if there is validation to do
        if early_stopper is None:
            return False

        # We put the model in eval mode
        self.eval()

        # We execute a forward pass and compute the loss and the score on the valid set
        valid_idx = dataset.valid_mask
        sample_weights = ones(len(valid_idx))/len(valid_idx)  # Equal sample weights for valid (1/n)
        with no_grad():

            output = self(dataset.graph, dataset.x_cont)
            loss = self.loss(sample_weights, output[valid_idx], dataset.y[valid_idx])
            score = self.eval_metric(output[valid_idx])

        # We save the loss and the score
        self._evaluations["valid"][self._criterion_name].append(loss)
        self._evaluations["valid"][self._eval_metric.name].append(score)

        # We check early stopping status
        early_stopper(loss, self)

        if early_stopper.early_stop:
            self.load_state_dict(early_stopper.get_best_params())
            return True

        return False

    def forward(self, g: dgl.DGLHeteroGraph, h: tensor):

        # We make a forward pass through han layers
        for gnn in self.gnn_layers:
            h = gnn(g, h)

        # We pass the final embedding through a linear layer
        return self.linear_layer(h)

    def fit(self, dataset: PetaleStaticGNNDataset, lr: float, max_epochs: int = 200, patience: int = 15,
            sample_weights: Optional[tensor] = None) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which stores an heterogeneous graph
            lr: learning rate
            max_epochs: Maximum number of epochs for training
            patience: Number of consecutive epochs without improvement
            sample_weights: (N,) tensor with weights of the samples in the training set

        Returns: None
        """
        # We check if there is a validation set
        if len(dataset.valid_mask) != 0:
            early_stopper, early_stopping = EarlyStopper(patience), True
        else:
            early_stopper, early_stopping = None, False

        # We initialize the optimizer
        self._optimizer = Adam(self.parameters(), lr=lr)

        # We init the update function
        update_progress = self._generate_progress_func(max_epochs)

        # We execute the training loop
        for epoch in range(max_epochs):

            # We execute a training step
            loss = self._execute_train_step(dataset, sample_weights)
            update_progress(epoch, loss)

            # We execute a validation step and check for early stopping
            if self._execute_valid_step(dataset, early_stopper):
                break

        if early_stopping:
            early_stopper.remove_checkpoint()



