"""
Author: Nicolas Raymond

This file is used to store the base skeleton of custom pytorch models
"""

from src.utils.score_metrics import Metric
from torch import tensor, mean, zeros_like
from torch.nn import Module
from torch.nn.functional import l1_loss, mse_loss
from typing import Callable


class TorchCustomModel(Module):
    """
    Use to store common protected attribute of torch custom models
    and loss function with elastic net penalty
    """
    def __init__(self, criterion: Callable, criterion_name: str, eval_metric: Metric,
                 alpha: float = 0, beta: float = 0, verbose: bool = False):
        """
        Sets protected attributes

        Args:
            criterion: loss function of our model
            criterion_name: name of the loss function of our model
            eval_metric: name of the loss function of our model
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            verbose: True if we want trace of the training progress
        """
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._criterion = criterion
        self._criterion_name = criterion_name
        self._eval_metric = eval_metric
        self._evaluations = {i: {self._criterion_name: [], self._eval_metric.name: []} for i in ["train", "valid"]}
        self._optimizer = None
        self._verbose = verbose

    def _generate_progress_func(self, max_epochs: int) -> Callable:
        """
        Defines a function that updates the training progress

        Args:
            max_epochs: maximum number of training epochs

        Returns: function
        """
        if self._verbose:
            def update_progress(epoch: int, mean_epoch_loss: float):
                if (epoch + 1) % 5 == 0 or (epoch + 1) == max_epochs:
                    print(f"Epoch {epoch + 1} - Loss : {round(mean_epoch_loss, 4)}")
        else:
            def update_progress(*args):
                pass

        return update_progress

    def loss(self, sample_weights: tensor, pred: tensor, y: tensor) -> tensor:
        """
        Calls the criterion and add elastic penalty

        Args:
            sample_weights: (N,) tensor with weights of samples on which we calculate loss
            pred: (N, C) tensor if classification with C classes, (N,) tensor for regression
            y: (N,) tensor with targets

        Returns: tensor with loss value
        """
        # Computations of penalties
        flatten_params = [w.view(-1, 1) for w in self.parameters()]
        l1_penalty = mean(tensor([l1_loss(w, zeros_like(w)) for w in flatten_params]))
        l2_penalty = mean(tensor([mse_loss(w, zeros_like(w)) for w in flatten_params]))

        # Computation of loss without reduction
        loss = self._criterion(pred, y.float())  # (N,) tensor

        # Computation of loss reduction + elastic penalty
        return (loss * sample_weights / sample_weights.sum()).sum() + self._alpha * l1_penalty + self._beta * l2_penalty