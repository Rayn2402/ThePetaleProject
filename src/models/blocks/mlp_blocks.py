"""
Filename: mlp_blocks.py

Author: Nicolas Raymond

Description: Defines neural network architecture blocks mostly related to MLPs

Date of last modification: 2021/11/08
"""

import torch.nn as nn
from torch import cat, tensor
from typing import List


class EntityEmbeddingBlock(nn.Module):
    """
    Contains a list of entity embedding layers associated to different categorical features
    """
    def __init__(self,
                 cat_sizes: List[int],
                 cat_emb_sizes: List[int],
                 cat_idx: List[int]):
        """
        Creates a ModuleList with the embedding layers

        Args:
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            cat_idx: list of idx associated to categorical columns in the dataset
        """
        # Call of parent's constructor
        super().__init__()

        # We save the length of the output
        self.__output_length = sum(cat_emb_sizes)

        # We save the idx of categorical columns
        self.__cat_idx = cat_idx

        # We create the ModuleList
        self.__embedding_layers = nn.ModuleList([nn.Embedding(cat_size, emb_size) for
                                                 cat_size, emb_size in zip(cat_sizes, cat_emb_sizes)])

    def __len__(self):
        """
        Returns the length of all embeddings concatenated
        """
        return self.__output_length

    def forward(self, x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: (N, D') tensor with concatenated embedding
        """
        # We calculate the embeddings
        x_cat = x[:, self.__cat_idx]
        embeddings = [e(x_cat[:, i].long()) for i, e in enumerate(self.__embedding_layers)]

        # We concatenate all the embeddings
        return cat(embeddings, 1)


class BaseBlock(nn.Module):
    """
    Linear -> BatchNorm -> Activation -> Dropout
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str,
                 p: float = 0):
        """
        Sets the layer attribute

        Args:
            input_size: size of input
            output_size: size of output
            activation: name of the activation function
            p: dropout probability
        """
        # Call of parent's constructor
        super().__init__()

        # We create a list with the modules
        module_list = [nn.Linear(input_size, output_size),
                       getattr(nn, activation)(),
                       nn.BatchNorm1d(output_size)]
        if p != 0:
            module_list.append(nn.Dropout(p))

        # We create a sequential from the list
        self.__layer = nn.Sequential(*module_list)

    def forward(self, x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: (N, D') tensor with concatenated embedding
        """
        self.__layer(x).squeeze()

