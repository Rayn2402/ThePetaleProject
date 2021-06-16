"""
Authors : Mitiche

File that contains class related to the Early Stopping

"""

import numpy as np


from os import path, makedirs, remove
from settings.paths import Paths
from torch import save, load
from torch.nn import Module
from uuid import uuid4


class EarlyStopping:

    def __init__(self, patience: int):

        """
        Creates a class that will be responsible of the early stopping when training the model

        :param patience: int representing how long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.early_stop = False
        self.counter = 0
        self.best_model = None
        self.val_loss_min = np.inf
        self.file_path = path.join(Paths.CHECKPOINTS, f"{uuid4()}.pt")

    def __call__(self, val_loss: float, model: Module) -> None:
        """
        Method called to perform the early stopping logic
        """

        # if the score is worst than the best score we increment the counter
        if val_loss > self.val_loss_min:
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                self.early_stop = True

        # if the score is better than the best score saved we update the best model
        else:
            self.val_loss_min = val_loss
            save(model, self.file_path)
            self.counter = 0

    def remove_checkpoint(self):
        """
        Removes the checkpoint file
        """
        remove(self.file_path)

    def get_best_model(self) -> Module:
        """
        Returns the best model saved

        :return: nn.Module
        """
        return load(self.file_path)
