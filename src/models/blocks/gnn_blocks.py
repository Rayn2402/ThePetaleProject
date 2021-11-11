"""
Filename: gnn_blocks.py

Author: Nicolas Raymond

Description: Defines neural network architecture blocks mostly related to GNNs

Date of last modification: 2021/11/09
"""

from dgl import DGLHeteroGraph, metapath_reachable_graph
from dgl.nn.pytorch import GATConv
from torch import stack, tensor
from torch.nn import Linear, Module, ModuleList, Sequential, Tanh
from torch.nn.functional import softmax, elu
from typing import List, Union


class HANLayer(Module):
    """
    Represents a single layer of the Heterogeneous Graph Attention Network model.

    A layers is composed of :
        - One Graph Attention Network for each meta path based graph
        - An additional attention layer to capture meta paths' importance
          and concatenate embedding learnt from every meta paths.
    """
    def __init__(self,
                 meta_paths: List[Union[str, List[str]]],
                 in_size: int,
                 out_size: int,
                 layer_num_heads: int,
                 dropout: float):
        """
        Builds a single layer of the Heterogeneous Graph Attention Network model.

        Args:
            meta_paths: list of metapaths, each meta path is a list of edge types or a string of a single edge type
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
            self.gat_layers.append(GATConv(in_feats=in_size,
                                           out_feats=out_size,
                                           num_heads=layer_num_heads,
                                           feat_drop=dropout,
                                           attn_drop=dropout,
                                           activation=elu,
                                           allow_zero_in_degree=True))

        # Semantic attention layer
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)

        # Meta paths list
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        # Cached protected attributes
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self,
                g: DGLHeteroGraph,
                h: tensor) -> tensor:

        # We initialize the storage for semantic embeddings
        semantic_embeddings = []

        # We create a cache of sub graphs associated to each meta path
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()

            # For each metapath
            for meta_path in self.meta_paths:

                # We extract the homogenous graph associated
                homogeneous_g = metapath_reachable_graph(g, meta_path)

                # We make sure that each node has a single self-loop
                homogeneous_g.remove_self_loop()
                homogeneous_g.add_self_loop()

                self._cached_coalesced_graph[meta_path] = homogeneous_g

        # For each meta path
        for i, meta_path in enumerate(self.meta_paths):

            # We extract the homogeneous graph associated to the metapath
            new_g = self._cached_coalesced_graph[meta_path]

            # We proceed to a forward pass in a GAT
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))

        # We stack the embeddings learnt using each meta path neighborhood
        semantic_embeddings = stack(semantic_embeddings, dim=1)                        # (N, M, D * K)

        # We pass these embeddings through an attention layer
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)


class SemanticAttention(Module):
    """
    Attention layer that identifies meta paths' importance in HAN
    """
    def __init__(self,
                 in_size: int,
                 hidden_size: int = 128):
        """
        Initializes the layers needed to execute the projection from which
        the semantic attention coefficients are calculated

        Args:
            in_size: input size of embeddings learned with node level attention
            hidden_size: size of the linear projection used to compute semantic attention
        """
        super(SemanticAttention, self).__init__()

        # Projection = q_t (tanh(Wz + b)) where q is a vector and W is a matrix
        self.project = Sequential(
            Linear(in_size, hidden_size),
            Tanh(),
            Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z: tensor) -> tensor:
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
