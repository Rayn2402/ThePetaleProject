"""
Author : Nicolas Raymond

This file contains metric used to measure models' performances
"""
from torch import sqrt, abs, tensor, argmax, zeros, unique
from torch.nn.functional import cross_entropy
REDUCTIONS = ["mean", "geometric_mean"]

class RegressionMetrics:

    @staticmethod
    def pearson(pred: tensor, targets: tensor) -> float:
        """
        Computes the pearson correlation coefficient between predictions and targets
        NOTE! : A strong correlation does not imply good accuracy

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: float
        """
        p = pred - pred.mean()
        t = targets - targets.mean()

        return (p.dot(t) / (sqrt((p**2).sum())*sqrt((t**2).sum()))).item()

    @staticmethod
    def mean_absolute_error(pred: tensor, targets: tensor) -> float:
        """
        Computes the mean absolute error between pred and targets

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: float
        """
        return abs(pred - targets).mean().item()


class ClassificationMetrics:

    @staticmethod
    def accuracy(pred: tensor, targets: tensor) -> float:
        """
        Returns the accuracy of predictions

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: float
        """
        return (argmax(pred, dim=1).float() == targets).float().mean().item()

    @staticmethod
    def cross_entropy_loss(pred: tensor, targets: tensor) -> float:
        """
        Returns the cross entropy related to predictions

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: float
        """

        return cross_entropy(pred, targets.long()).item()

    @staticmethod
    def acc_cross(pred: tensor, targets: tensor) -> float:
        """
        Returns the ratio accuracy/cross-entropy related to predictions

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: float
        """
        return ClassificationMetrics.accuracy(pred, targets)/ClassificationMetrics.cross_entropy_loss(pred, targets)

    @staticmethod
    def get_confusion_matrix(pred: tensor, targets: tensor) -> tensor:
        """
        Returns the confusion matrix

        :param pred: (N,) tensor
        :param targets: (N,) tensor
        :return: (C,C) tensor
        """

        pred = argmax(pred, 1)
        targets = targets.long()

        nb_classes = unique(targets).shape[0]
        conf_matrix = zeros(nb_classes, nb_classes)
        for t, p in zip(targets, pred):
            conf_matrix[t, p] += 1

        return conf_matrix

