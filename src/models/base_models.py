"""
Author: Nicolas Raymond

This file defines the abstract Regressor and Classifier classes that must be used
to build every other model in the project
"""
from abc import ABC, abstractmethod
from numpy import array
from numpy import where as npwhere
from numpy import zeros as npzeros
from src.data.processing.datasets import PetaleDataset
from torch import tensor, is_tensor
from torch import where as thwhere
from torch import zeros as thzeros
from typing import Optional, Union


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

    def get_sample_weights(self, y_train: Union[tensor, array]) -> Union[tensor, array]:
        """
        Computes the weights to give to each sample

        We need to solve the following equation:
            - n1 * w1 = self.weight
            - n0 * w0 = 1 - self.weight

        where n0 is the number of samples with label 0
        and n1 is the number of samples with label 1

        Args:
            y_train: (N, 1) tensor or array with labels

        Returns: sample weights
        """
        # If no weight was provided we return None
        if self.weight is None:
            return None

        # Otherwise we return samples' weights in the appropriate format
        n1 = y_train.sum()              # number of samples with label 1
        n0 = y_train.shape[0] - n1      # number of samples with label 0
        w0, w1 = (1 - self.weight) / n0, self.weight / n1  # sample weight for C0, sample weight for C1

        if not is_tensor(y_train):
            sample_weights = npzeros(y_train.shape)
            sample_weights[npwhere(y_train == 0)] = w0 * 10  # Multiply by constant factor to impact learning rate
            sample_weights[npwhere(y_train == 1)] = w1 * 10

        else:
            sample_weights = thzeros(y_train.shape)
            sample_weights[thwhere(y_train == 0)] = w0 * 10  # Multiply by constant factor to impact learning rate
            sample_weights[thwhere(y_train == 1)] = w1 * 10

        return sample_weights

    @abstractmethod
    def fit(self, dataset: PetaleDataset, **kwargs) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, dataset: PetaleDataset) -> Union[tensor, array]:
        """
        Returns the probabilities of being in class 1 for all samples
        in the test set

        Args:
            dataset: PetaleDatasets which items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: (N,) tensor or array
        """
        raise NotImplementedError


class PetaleRegressor(ABC):
    """
    Skeleton of all Petale regression models
    """
    @abstractmethod
    def fit(self, dataset: PetaleDataset, **kwargs) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which items are tuples (x, y) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: PetaleDataset) -> Union[tensor, array]:
        """
        Returns the predicted real-valued targets for all samples in the test set

        Args:
            dataset: PetaleDatasets which items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: (N,) tensor or array
        """
        raise NotImplementedError
