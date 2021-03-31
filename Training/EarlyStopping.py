"""
Authors : Mitiche

File that contains class related to the Early Stopping

"""

import numpy as np
from torch import save


class EarlyStopping:
    def __init__(self, patience):
        """
        Creates a class that will be responsible of the early stopping when training the model

        :param patience: int representing how long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.early_stop = False
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        """
        Method to be called to perform the early stopping logic
        """
        score = -val_loss




        # if there is not best score yet we save the score and the model
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # if the score is worst than the best score we increment the counter
        elif score < self.best_score:
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                self.early_stop = True

        # if the score is better than the best score we save the score and the model
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        method that will save the best model and store the validation loss of that model
        
        :param val_loss: the valid loss of the model to save.
        :param model: the model to save.
        """
        save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss
