"""
Author : Nicolas Raymond

This file contains metric used to measure models' performances
"""
from torch import sqrt, abs


class RegressionMetrics:

    @staticmethod
    def pearson(pred, targets):
        """
        Computes the pearson correlation coefficient between predictions and targets
        NOTE! : A strong correlation does not imply good accuracy

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: (1,) tensor
        """
        p = pred - pred.mean()
        t = targets - targets.mean()

        return p.dot(t) / (sqrt((p**2).sum())*sqrt((t**2).sum()))

    @staticmethod
    def mean_absolute_error(pred, targets):
        """
        Computes the mean absolute error between pred and targets

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: (1,) tensor
        """
        return abs(pred - targets).mean()


class ClassificationMetrics:

    @staticmethod
    def accuracy(pred, targets):
        """
        Returns the accuracy of predictions

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: (1,) tensor
        """
        return (pred == targets).float().mean()
