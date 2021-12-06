"""
Filename: encoder.py

Author: Nicolas Raymond

Description: Stores the Encoder abstract class

Date of last modification: 2021/12/06
"""

from abc import ABC


class Encoder(ABC):
    """
    Abstract class for encoder
    """
    def __init__(self,
                 input_size: int,
                 output_size: int):
        """
        Saves the input size and the output size

        Args:
            input_size: number of input features
            output_size: number of features in the encodings
        """
        self._input_size = input_size
        self._output_size = output_size

    @property
    def output_size(self):
        return self._output_size
