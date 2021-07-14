"""
Author: Nicolas Raymond

This file defines the abstract Regressor and Classifier classes that must be used
to build every other model in the project
"""
from abc import ABC, abstractmethod
from numpy import array
from torch import tensor, is_tensor, from_numpy
from typing import List, Optional, Tuple, Union


class PetaleBinaryClassifier(ABC):
    """
    Skeleton of all Petale classification models
    """
    def __init__(self, classification_threshold: float = 0.5, class_weights: Optional[List[float]] = None):
        """
        Sets the threshold for binary classification

        Args:
            classification_threshold: threshold used to classify a sample in class 1
            class_weights: weights attributes to samples of each class
        """
        self._thresh = classification_threshold
        self._class_weights = class_weights

    @property
    def thresh(self) -> float:
        return self._thresh

    @property
    def class_weights(self) -> Optional[float]:
        return self._class_weights

    def predict(self, x: Union[tensor, array]) -> tensor:
        """
        Returns the predicted binary classes associated to samples

        Args:
            x: (N,D) tensor or array with D-dimensional samples

        Returns: (N,) tensor
        """
        proba = self.predict_proba(x)
        if not is_tensor(proba):
            proba = from_numpy(proba).float()

        return (proba >= self._thresh).long()

    @abstractmethod
    def fit(self, x_train: Union[tensor, array], y_train: Union[tensor, array],
            eval_set: Optional[Tuple[Union[tensor, array], Union[tensor, array]]] = None, **kwargs) -> None:
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
            eval_set: Optional[Tuple[Union[tensor, array], Union[tensor, array]]] = None, **kwargs) -> None:
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
    def predict(self, x: Union[tensor, array]) -> tensor:
        """
        Returns the predicted real-valued targets for all samples

        Args:
            x: (N,D) tensor or array with D-dimensional samples

        Returns: (N,) tensor
        """
        raise NotImplementedError
