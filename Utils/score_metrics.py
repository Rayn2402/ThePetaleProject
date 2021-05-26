"""
Author : Nicolas Raymond

This file contains metric used to measure models' performances
"""
from abc import ABC, abstractmethod
from torch import sqrt, abs, tensor, argmax, zeros, unique, ones, eye, mean, prod, sum, pow
from torch.nn.functional import nll_loss

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
    def __init__(self, direction: str, task_type: str):
        """
        Sets protected attributes

        Args:
            direction: "maximize" or "minimize"
            task_type: "regression" or "classification"
        """
        # We call super init since we're using ABC
        super().__init__()

        # Protected attributes
        self._direction = direction
        self._task_type = task_type

    @property
    def direction(self):
        return self._direction

    @property
    def task_type(self):
        return self._task_type

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
    def __init__(self):
        super().__init__(MAXIMIZE, REG)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Computes the pearson correlation coefficient between predictions and targets
        NOTE! : A strong correlation does not imply good accuracy
        Args:
            pred: (N,) tensor
            targets: (N,) tensor

        Returns: float

        """
        p = pred - pred.mean()
        t = targets - targets.mean()

        return (p.dot(t) / (sqrt((p**2).sum())*sqrt((t**2).sum()))).item()


class AbsoluteError(Metric):
    """
    Callable class that computes the absolute error
    """
    def __init__(self, reduction: str = "mean"):
        """
        Sets the protected reduction method

        Args:
            reduction: "mean" for mean absolute error and "sum" for the sum of the absolute errors
        """
        assert reduction in [MEAN, SUM], f"Reduction must be in {[MEAN, SUM]}"

        super().__init__(MINIMIZE, REG)
        self._reduction = mean if "mean" else sum

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Compute the absolute error between predictions and targets

        Args:
            pred: (N,) tensor
            targets: (N,) tensor

        Returns: float

        """
        return self._reduction(abs(pred - targets)).item()


class Accuracy(Metric):
    """
    Callable class that computes the accuracy
    """
    def __init__(self):
        super().__init__(MAXIMIZE, CLASSIFICATION)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Returns the accuracy of predictions

        Args:
            pred: (N,C) tensor with log probabilities
            targets: (N,) tensor

        Returns: float

        """
        return (argmax(pred, dim=1).float() == targets.long()).float().mean().item()


class CrossEntropyLoss(Metric):
    """
    Callable class that computes the cross entropy loss
    """
    def __init__(self, reduction: str = "mean"):
        """
        Sets the protected reduction method
        Args:
            reduction: "mean" for mean cross entropy and "sum" for the sum of the cross entropy losses
        """
        super().__init__(MINIMIZE, CLASSIFICATION)
        self._reduction = reduction

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Returns the cross entropy related to the predictions

        Args:
            pred: (N,C) tensor with log probabilities
            targets: (N,) tensor

        Returns: float
        """
        return nll_loss(pred, targets.long(), reduction=self._reduction).item()


class AccuracyCrossEntropyRatio(Metric):
    """
    Callable class that computes the ratio accuracy/cross_entropy
    """

    def __init__(self, reduction: str = "mean"):
        """
        Sets the protected callable attributes accuracy and cross entropy

        Args:
            reduction: "mean" for mean cross entropy and "sum" for the sum of the cross entropy losses
        """
        super().__init__(MAXIMIZE, CLASSIFICATION)
        self._accuracy = Accuracy()
        self._cross_entropy = CrossEntropyLoss(reduction=reduction)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Returns the ratio accuracy/cross-entropy related to the predictions

        Args:
            pred: (N,C) tensor with log probabilities
            targets: (N,) tensor

        Returns: float

        """
        return self._accuracy(pred, targets)/self._cross_entropy(pred, targets)


class Sensitivity(Metric):
    """
    Callable class that compute classes' sensitivity using confusion matrix
    """
    def __init__(self, reduction: str = "mean"):
        """
        Sets the protected reduction method

        Args:
            reduction: "mean" for mean classes' sensitivity or "geometric_mean"
                       for geometric mean of classes' sensitivity
        """
        assert reduction in [MEAN, GEO_MEAN], f"Reduction must be in {[MEAN, GEO_MEAN]}"
        super().__init__(MAXIMIZE, CLASSIFICATION)

        if reduction == "mean":
            self._reduction = mean
        else:
            self._reduction = lambda x: pow(prod(x), exponent=x.shape[0])

    @staticmethod
    def get_confusion_matrix(pred: tensor, targets: tensor) -> tensor:
        """
        Returns the confusion matrix

        Args:
            pred: (N,C) tensor with log probabilities
            targets: (N,) tensor

        Returns: (C,C) tensor

        """

        # We extract the class prediction and convert targets to long type
        pred = argmax(pred, 1)
        targets = targets.long()

        # We extract the number of possible class and initialize and empty confusion matrix
        nb_classes = unique(targets).shape[0]
        conf_matrix = zeros(nb_classes, nb_classes)

        # We fill the confusion matrix
        for t, p in zip(targets, pred):
            conf_matrix[t, p] += 1

        return conf_matrix

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Returns the classes' sensitivity

        Args:
            pred: (N,C) tensor with log probabilities
            targets: (N,) tensor

        Returns: float
        """

        # We get confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets)

        # We first get true positives
        tp = conf_mat.diag()

        # We then get false negatives (row sums of items of diagonal)
        fn = (conf_mat * (ones(conf_mat.shape)-eye(conf_mat.shape[0]))).sum(axis=1)

        # We compute class sensitivity
        sensitivity = tp / (tp+fn)

        return self._reduction(sensitivity).item()


class SensitivityCrossEntropyRatio(Metric):
    """
    Callable class that computes the sensitivity/cross_entropy ratio
    """
    def __init__(self, reduction: str = "mean"):
        """
        Sets the sensitivity and cross entropy protected attributes

        Args:
            reduction: "mean" for mean classes' sensitivity or "geometric_mean"
                       for geometric mean of classes' sensitivity
        """
        super().__init__(MAXIMIZE, CLASSIFICATION)
        self._sensitivity = Sensitivity(reduction)
        self._cross_entropy = CrossEntropyLoss(reduction)

    def __call__(self, pred: tensor, targets: tensor) -> float:
        """
        Returns the sensitivity/cross_entropy ratio

        Args:
            pred: (N,C) tensor with log probabilities
            targets: (N,) tensor

        Returns: float
        """
        return self._sensitivity(pred, targets)/self._cross_entropy(pred, targets)
