"""
Filename: gnn_blocks.py

Author: Nicolas Raymond

Description: Defines neural network architecture blocks mostly related to GNNs

Date of last modification: 2022/04/12
"""

from torch import bernoulli, ones, tensor
from torch.nn import Module


class DropNode(Module):
    """
    Node Dropout layer

    Base on drop_node function implemented by DGL:
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/grand/model.py

    """
    def __init__(self, p: float = 0.5):
        """
        Sets the dropout probability

        Args:
            p: dropout probability
        """
        super().__init__()
        self._p = p

    def forward(self, x: tensor) -> tensor:
        """
        Sets randomly rows of the tensor to zero

        Args:
            x: (N, D) tensor with D-dimensional samples

        Returns: (N, D) tensor
        """
        drop_rates = ones(x.shape[0]) * self._p
        if self.training:

            masks = bernoulli(1. - drop_rates).unsqueeze(1)
            x = masks * x

        else:
            x = x * (1. - drop_rates.unsqueeze(1))

        return x


