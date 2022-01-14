"""
Filename: early_stopping.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to define the EarlyStopper class

Date of last modification : 2021/10/29
"""

import numpy as np

from os import path, remove
from settings.paths import Paths
from src.utils.score_metrics import Direction
from torch import load, save, tensor
from torch.nn import Module
from typing import OrderedDict
from uuid import uuid4


class EarlyStopper:

    def __init__(self,
                 patience: int,
                 direction: str):
        """
        Sets protected attributes of early stopper and define comparison
        method according to the given direction

        Args:
            patience: number of epochs without improvement
            direction: "minimize" or "maximize"
        """
        # Set public attribute
        self.patience = patience
        self.early_stop = False
        self.counter = 0
        self.best_model = None
        self.file_path = path.join(Paths.CHECKPOINTS, f"{uuid4()}.pt")

        # Set comparison method
        if direction == Direction.MINIMIZE:
            self.best_val_score = np.inf
            self.is_better = lambda x, y: x < y

        elif direction == Direction.MAXIMIZE:
            self.best_val_score = -np.inf
            self.is_better = lambda x, y: x > y
        else:
            raise ValueError(f'direction must be in {list(Direction())}')

    def __call__(self,
                 val_score: float,
                 model: Module) -> None:
        """
        Compares current best validation score against the given one and updates
        the EarlyStopper's attributes

        Args:
            val_score: new validation score
            model: current model for which we've seen the score

        Returns: None
        """
        # if the score is worst than the best score we increment the counter
        if not self.is_better(val_score, self.best_val_score):
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                self.early_stop = True

        # if the score is better than the best score saved we update the best model
        else:
            self.best_val_score = val_score
            save(model.state_dict(), self.file_path)
            self.counter = 0

    def remove_checkpoint(self) -> None:
        """
        Removes the checkpoint file
        """
        remove(self.file_path)

    def get_best_params(self) -> OrderedDict[str, tensor]:
        """
        Returns the best parameters saved

        :return: model state dict
        """
        return load(self.file_path)
