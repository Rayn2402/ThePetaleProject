"""
Author: Nicolas Raymond

This file defines the abstract Regressor and Classifier classes that must be used
to build every other model in the project
"""
from abc import ABC, abstractmethod
from numpy import array
from torch import tensor
from typing import Optional, Tuple, Union


class PetaleBinaryClassifier(ABC):
    """
    Skeleton of all Petale classification models
    """
    def __init__(self, classification_threshold: float = 0.5, weight:  Optional[float] = None):
        """
        Sets the threshold for binary classification

        Args:
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        if weight is not None:
            assert 0 <= weight <= 1, "weight must be included in range [0, 1]"

        self._thresh = classification_threshold
        self._weight = weight

    @property
    def thresh(self) -> float:
        return self._thresh

    @property
    def weight(self) -> Optional[float]:
        return self._weight

    def get_sample_weights(self, n0: int, n1: int) -> Tuple[float, float]:
        """
        Computes the weights to give to each sample of each class
        We need to solve the following equation:
            - n1 * w1 = self.weight
            - n0 * w0 = 1 - self.weight
        Args:
            n0: number of samples with label 0
            n1: number of samples with label 1

        Returns: sample weight for C0, sample weight for C1
        """
        return (1 - self.weight)/n0, self.weight/n1

    @abstractmethod
    def fit(self, x_train: Union[tensor, array], y_train: Union[tensor, array],
            eval_set: Optional[Tuple[Union[tensor, array], Union[tensor, array]]] = None) -> None:
        """
        Fits the model to the training data

        Args:
            x_train: (N,D) tensor or array with D-dimensional samples
            y_train: (N,1) tensor or array with classification labels
            eval_set: Tuple with validation set

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: Union[tensor, array]) -> Union[tensor, array]:
        """
        Returns the probabilities of being in class 1 for all samples

        Args:
            x: (N,D) tensor or array with D-dimensional samples

        Returns: (N,) tensor or array
        """
        raise NotImplementedError


class PetaleRegressor(ABC):
    """
    Skeleton of all Petale regression models
    """
    @abstractmethod
    def fit(self, x_train: Union[tensor, array], y_train: Union[tensor, array],
            eval_set: Optional[Tuple[Union[tensor, array], Union[tensor, array]]] = None) -> None:
        """
        Fits the model to the training data

        Args:
            x_train: (N,D) tensor or array with D-dimensional samples
            y_train: (N,1) tensor or array with classification labels
            eval_set: Tuple with validation set

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Union[tensor, array]) -> Union[tensor, array]:
        """
        Returns the predicted real-valued targets for all samples

        Args:
            x: (N,D) tensor or array with D-dimensional samples

        Returns: (N,) tensor or array
        """
        raise NotImplementedError
