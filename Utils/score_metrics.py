"""
Author : Nicolas Raymond

This file contains metric used to measure models' performances
"""
from torch import sqrt, abs, tensor, argmax
from torch.nn.functional import cross_entropy


class RegressionMetrics:

    @staticmethod
    def pearson(pred: tensor, targets: tensor) -> tensor:
        """
        Computes the pearson correlation coefficient between predictions and targets
        NOTE! : A strong correlation does not imply good accuracy

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: (1,) tensor
        """
        p = pred - pred.mean()
        t = targets - targets.mean()

        return (p.dot(t) / (sqrt((p**2).sum())*sqrt((t**2).sum()))).item()

    @staticmethod
    def mean_absolute_error(pred: tensor, targets: tensor) -> tensor:
        """
        Computes the mean absolute error between pred and targets

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: (1,) tensor
        """
        return abs(pred - targets).mean().item()


class ClassificationMetrics:

    @staticmethod
    def accuracy(pred: tensor, targets: tensor) -> tensor:
        """
        Returns the accuracy of predictions

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: (1,) tensor
        """
        return (argmax(pred, dim=1).float() == targets).float().mean().item()

    @staticmethod
    def cross_entropy_loss(pred, target):
        """
        Returns the cross entropy related to predictions

        :param pred: (N,) tensor
        :param target: (N,) tensor
        :return: (1,) tensor
        """

        return cross_entropy(pred, target.long()).item()
