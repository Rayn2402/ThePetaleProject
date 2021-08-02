"""
Authors : Mitiche

File that contains class related to the Early Stopping

"""

import numpy as np


from os import path, remove
from settings.paths import Paths
from src.utils.score_metrics import Direction
from torch import save, load, tensor
from torch.nn import Module
from typing import OrderedDict
from uuid import uuid4


class EarlyStopper:

    def __init__(self, patience: int, direction: str):
        """
        Sets protected attributes of early stopper and define comparison
         method according to given direction

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
            self.val_score_min = np.inf
            self.is_better = lambda x, y: x < y
        else:
            self.val_score_min = -np.inf
            self.is_better = lambda x, y: x > y

    def __call__(self, val_score: float, model: Module) -> None:
        """
        Method called to perform the early stopping logic
        """

        # if the score is worst than the best score we increment the counter
        if not self.is_better(val_score, self.val_score_min):
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                self.early_stop = True

        # if the score is better than the best score saved we update the best model
        else:
            self.val_score_min = val_score
            save(model.state_dict(), self.file_path)
            self.counter = 0

    def remove_checkpoint(self):
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
