"""
Author : Nicolas Raymond

This file contains metric used to measure models' performances
"""
from torch import sqrt, abs, tensor, argmax, zeros, unique, ones, eye, mean, prod
from torch.nn.functional import nll_loss
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

        :param pred: (N,C) tensor with log probabilities
        :param targets: (N,) tensor
        :return: float
        """
        return (argmax(pred, dim=1).float() == targets).float().mean().item()

    @staticmethod
    def cross_entropy_loss(pred: tensor, targets: tensor) -> float:
        """
        Returns the cross entropy related to the predictions

        :param pred: (N,C) tensor with log probabilities
        :param targets: (N,) tensor
        :return: float
        """

        return nll_loss(pred, targets.long()).item()

    @staticmethod
    def acc_cross(pred: tensor, targets: tensor) -> float:
        """
        Returns the ratio accuracy/cross-entropy related to the predictions

        :param pred: (N,C) tensor with log probabilities
        :param targets: (N,) tensor
        :return: float
        """
        acc = ClassificationMetrics.accuracy(pred, targets)
        cross = ClassificationMetrics.cross_entropy_loss(pred, targets)
        return acc/cross

    @staticmethod
    def sensitivity_cross(pred: tensor, targets: tensor, reduction: str = 'mean') -> float:
        """
        Returns the ratio (mean class sensitivity / cross entropy)

        :param pred: (N,C) tensor with log probabilities
        :param targets: (N,) tensor
        :param reduction: str
        :return: float
        """
        sensitivity = ClassificationMetrics.class_sensitivity(pred, targets, reduction)
        cross = ClassificationMetrics.cross_entropy_loss(pred, targets)
        return sensitivity/cross

    @staticmethod
    def class_sensitivity(pred: tensor, targets: tensor, reduction: str = 'mean') -> float:
        """
        Returns the mean classes sensitivity

        :param pred: (N,C) tensor with log probabilities
        :param targets: (N,) tensor
        :param reduction: str
        :return: float
        """

        assert reduction in REDUCTIONS, "Reduction selected not available"

        # We get confusion matrix
        confusion_mat = ClassificationMetrics.get_confusion_matrix(pred, targets)

        # We first get true positives
        TP = confusion_mat.diag()

        # We then get false negatives (row sums of items of diagonal)
        FN = (confusion_mat * (ones(confusion_mat.shape)-eye(confusion_mat.shape[0]))).sum(axis=1)

        # We compute class sensitivity
        sensitivity = TP / (TP+FN)

        # We return the mean of classes' sensitivity
        if reduction == 'mean':
            return mean(sensitivity).item()

        # We return the geometric mean of classes' sensitivity
        else:
            return (prod(sensitivity).item())**(1/sensitivity.shape[0])

    @staticmethod
    def get_confusion_matrix(pred: tensor, targets: tensor) -> tensor:
        """
        Returns the confusion matrix

        :param pred: (N,C) tensor with log probabilities
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

