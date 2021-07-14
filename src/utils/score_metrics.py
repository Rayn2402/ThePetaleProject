"""
Author : Nicolas Raymond

This file contains metric used to measure models' performances
"""
from abc import ABC, abstractmethod
from torch import sqrt, abs, tensor, zeros, mean, prod, sum, pow
from typing import Callable


MAXIMIZE = "maximize"
MINIMIZE = "minimize"
REG, CLASSIFICATION = "regression", "classification"
MEAN = "mean"
SUM = "sum"
GEO_MEAN = "geometric_mean"


class Metric(ABC):
    """
    Abstract class that represents the skeleton of callable classes to use as optimization metrics
    """
    def __init__(self, direction: str, name: str, task_type: str, n_digits: int = 5):
        """
        Sets protected attributes

        Args:
            direction: "maximize" or "minimize"
            name: name of the metric
            task_type: "regression" or "classification"
            n_digits: number of digits kept
        """
        # Protected attributes
        self._direction = direction
        self._name = name
        self._task_type = task_type
        self._n_digits = n_digits

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_digits(self) -> int:
        return self._n_digits

    @abstractmethod
    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Metric calculated by the class
        """
        raise NotImplementedError


class Pearson(Metric):
    """
    Callable class that computes Pearson correlation coefficient
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        super().__init__(MAXIMIZE, "Pearson", REG, n_digits)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Computes the pearson correlation coefficient between predictions and targets
        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth

        Returns: float

        """
        p = pred - pred.mean()
        t = targets - targets.mean()

        return round((p.dot(t) / (sqrt((p**2).sum())*sqrt((t**2).sum()))).item(), self.n_digits)


class AbsoluteError(Metric):
    """
    Callable class that computes the absolute error
    """
    def __init__(self, reduction: str = "mean", n_digits: int = 5):
        """
        Sets the protected reduction method and other protected attributes using parent's constructor

        Args:
            reduction: "mean" for mean absolute error and "sum" for the sum of the absolute errors
            n_digits: number of digits kept for the score
        """
        assert reduction in [MEAN, SUM], f"Reduction must be in {[MEAN, SUM]}"

        if reduction == "mean":
            name = "MAE"
            self._reduction = mean
        else:
            name = "AE"
            self._reduction = sum

        super().__init__(MINIMIZE, name, REG, n_digits)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Compute the absolute error between predictions and targets

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth

        Returns: float

        """
        return round(self._reduction(abs(pred - targets)).item(), self.n_digits)


class RootMeanSquaredError(Metric):
    """
    Callable class that computes root mean-squared error
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        super().__init__(MINIMIZE, "RMSE", REG, n_digits)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Compute the root mean-squared error between predictions and targets

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth

        Returns: float
        """
        return round((mean((pred - targets)**2).item())**(1/2), self.n_digits)


class BinaryAccuracy(Metric):
    """
    Callable class that computes the accuracy
    """
    def __init__(self, n_digits: int = 5):
        """
        Sets protected attributes using parent's constructor

        Args:
            n_digits: number of digits kept for the score
        """
        super().__init__(MAXIMIZE, "Accuracy", CLASSIFICATION, n_digits)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Returns the accuracy of predictions

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth

        Returns: float

        """
        return round((pred == targets).float().mean().item(), self.n_digits)


class BinaryBalancedAccuracy(Metric):
    """
    Callable class that compute classes' sensitivity using confusion matrix
    """
    def __init__(self, reduction: str = "mean", n_digits: int = 5):
        """
        Sets the protected reduction method

        Args:
            reduction: "mean" for mean classes' sensitivity or "geometric_mean"
                       for geometric mean of classes' sensitivity
        """
        assert reduction in [MEAN, GEO_MEAN], f"Reduction must be in {[MEAN, GEO_MEAN]}"

        if reduction == "mean":
            self._reduction = mean
            name = "BalancedAcc"
        else:
            self._reduction = lambda x: pow(prod(x), exponent=x.shape[0])
            name = "GeoBalancedAcc"

        super().__init__(MAXIMIZE, name, CLASSIFICATION, n_digits)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Returns the classes' sensitivity

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth

        Returns: float
        """
        # We get confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets)
        print(conf_mat)

        # We get TNR and TPR
        correct_rates = conf_mat.diag()/conf_mat.sum(dim=1)

        return round(self._reduction(correct_rates).item(), self.n_digits)

    @staticmethod
    def get_confusion_matrix(pred: tensor, targets: tensor) -> tensor:
        """
        Returns the confusion matrix

        Args:
            pred: (N,) tensor with predicted labels
            targets: (N,) tensor with ground truth

        Returns: (2,2) tensor

        """
        # We initialize an empty confusion matrix
        conf_matrix = zeros(2, 2)

        # We fill the confusion matrix
        for t, p in zip(targets, pred):
            conf_matrix[t, p] += 1

        return conf_matrix
