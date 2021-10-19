"""
Filename: base_models.py
Author: Nicolas Raymond
Description: Defines the abstract PetaleRegressor and PetaleBinaryClassifier
             classes that must be used to build every other model in the project.
             It ensures consistency will all hyperparameter tuning functions.

Date of last modification : 2021/10/19
"""
from abc import ABC, abstractmethod
from numpy import array, argmin, argmax, linspace
from numpy import where as npwhere
from numpy import zeros as npzeros
from src.data.processing.datasets import PetaleDataset
from src.utils.hyperparameters import HP
from src.utils.score_metrics import BinaryClassificationMetric, Direction
from torch import tensor, is_tensor
from torch import where as thwhere
from torch import zeros as thzeros
from typing import Any, Dict, List, Optional, Union


class PetaleBinaryClassifier(ABC):
    """
    Skeleton of all Petale binary classification models
    """
    def __init__(self, classification_threshold: float = 0.5,
                 weight:  Optional[float] = None,
                 train_params: Optional[Dict[str, Any]] = None):
        """
        Sets the protected attributes of the object

        Args:
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1 (in [0, 1])
            train_params: keyword arguments that are proper to the child model inheriting
                          from this class and that will be using when calling fit method
        """
        if weight is not None:
            if not (0 <= weight <= 1):
                raise ValueError("weight must be included in range [0, 1]")

        self._thresh = classification_threshold
        self._train_params = train_params if train_params is not None else {}
        self._weight = weight

    @property
    def thresh(self) -> float:
        return self._thresh

    @property
    def train_params(self) -> Dict[str, Any]:
        return self._train_params

    @property
    def weight(self) -> Optional[float]:
        return self._weight

    def find_optimal_threshold(self, dataset: PetaleDataset, metric: BinaryClassificationMetric) -> None:
        """
        Finds the optimal classification threshold for a binary classification task
        according to a given metric.

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

            metric: Binary classification metric used to find optimal threshold
        """
        # We predict proba on the training set
        proba = self.predict_proba(dataset, dataset.train_mask)

        # For multiple threshold values we calculate the metric
        thresholds = linspace(start=0.01, stop=0.95, num=95)
        scores = array([metric(proba, dataset.y[dataset.train_mask], t) for t in thresholds])

        # We save the optimal threshold
        if metric.direction == Direction.MINIMIZE:
            self._thresh = thresholds[argmin(scores)]
        else:
            self._thresh = thresholds[argmax(scores)]

    def get_sample_weights(self, y_train: Union[tensor, array]) -> Union[tensor, array]:
        """
        Computes the weight associated to each sample

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

        # We save the weights in the appropriate format and multiply them with
        # a constant to have an impact with low learning rate
        if not is_tensor(y_train):
            sample_weights = npzeros(y_train.shape)
            sample_weights[npwhere(y_train == 0)] = w0 * 10
            sample_weights[npwhere(y_train == 1)] = w1 * 10

        else:
            sample_weights = thzeros(y_train.shape)
            sample_weights[thwhere(y_train == 0)] = w0 * 10
            sample_weights[thwhere(y_train == 1)] = w1 * 10

        return sample_weights

    @staticmethod
    @abstractmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, dataset: PetaleDataset) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, dataset: PetaleDataset, mask: Optional[List[int]] = None) -> Union[tensor, array]:
        """
        Returns the probabilities of being in class 1 for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to predict proba

        Returns: (N,) tensor or array
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """
        raise NotImplementedError


class PetaleRegressor(ABC):
    """
    Skeleton of all Petale regression models
    """
    def __init__(self, train_params: Optional[Dict[str, Any]] = None):
        """
        Sets the only protected attribute

        Args:
            train_params: keyword arguments that are proper to the child model inheriting
                          from this class and that will be using when calling fit method
        """
        self._train_params = train_params if train_params is not None else {}

    @property
    def train_params(self) -> Dict[str, Any]:
        return self._train_params

    @staticmethod
    @abstractmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, dataset: PetaleDataset) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: PetaleDataset, mask: Optional[List[int]] = None) -> Union[tensor, array]:
        """
        Returns the predicted real-valued targets for all samples in a
        particular set (default = test)

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to make predictions

        Returns: (N,) tensor or array
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """
        raise NotImplementedError
