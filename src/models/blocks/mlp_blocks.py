"""
Filename: mlp_blocks.y

Author: Nicolas Raymond

Description: Defines neural network architecture blocks mostly related to MLPs

Date of last modification: 2021/11/08
"""

from torch.nn import Module, ModuleList
from typing import List


class EntityEmbeddingBlock(Module):
    """
    Contains a list of entity embedding layers associated to different categorical features
    """
    def __init__(self, cat_sizes: List[int], cat_emb_sizes: List[int]):
        """
        Creates a ModuleList with the embedding layers

        Args:
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
        """
        super().__init__()

        # We save the length of the output
        self._output_length = sum(cat_emb_sizes)

        # We create the ModuleList


