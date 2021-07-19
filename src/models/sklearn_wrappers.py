"""
Author: Nicolas Raymond

This file is used to store abstract class to use as wrappers for models with the sklearn API
"""
from numpy import array, where, zeros
from src.models.base_models import PetaleBinaryClassifier, PetaleRegressor
from typing import Callable, Optional, Tuple


class SklearnBinaryClassifierWrapper(PetaleBinaryClassifier):
    """
    Class used as a wrapper for binary classifier with sklearn API
    """
    def __init__(self, model: Callable, classification_threshold: int = 0.5,
                 weight: Optional[float] = None):

        """
        Sets model protected attribute and other protected attributes via parent's constructor

        Args:
            model: classification model with sklearn API
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        self._model = model
        super().__init__(classification_threshold=classification_threshold, weight=weight)

    def fit(self, x_train: array, y_train: array,
            eval_set: Optional[Tuple[array, array]] = None) -> None:
        """
        Fits the model to the training data

        Args:
            x_train: (N,D) array with D-dimensional samples
            y_train: (N,1) array with classification labels
            eval_set: Tuple with validation set

        Returns: None
        """
        # We set sample weights
        if self.weight is not None:
            sample_weights = zeros(y_train.shape)
            n1 = y_train.sum()
            n0 = y_train.shape[0] - n1
            w0, w1 = self.get_sample_weights(n0, n1)
            sample_weights[where(y_train == 0)] = w0*10
            sample_weights[where(y_train == 1)] = w1*10
        else:
            sample_weights = None

        # Call the sklearn fit method
        self._model.fit(x_train, y_train, sample_weight=sample_weights)

    def predict_proba(self, x: array) -> array:
        """
        Returns the probabilities of being in class 1 for all samples

        Args:
            x: (N,D) array with D-dimensional samples

        Returns: (N,) array
        """
        # Call sklearn predict_proba method, takes the prediction for class 1 and squeeze the array
        proba = self._model.predict_proba(x)[:, 1]

        return proba.squeeze()


class SklearnRegressorWrapper(PetaleRegressor):
    """
    Class used as a wrapper for regression model with sklearn API
    """
    def __init__(self, model: Callable):
        """
        Sets the model protected attribute

        Args:
            model: regression model with sklearn API
        """
        self._model = model

    def fit(self, x_train: array, y_train: array,
            eval_set: Optional[Tuple[array, array]] = None) -> None:
        """
        Fits the model to the training data

        Args:
            x_train: (N,D) array with D-dimensional samples
            y_train: (N,1) or array with classification labels
            eval_set: Tuple with validation set

        Returns: None
        """
        # Call the sklearn fit method
        self._model.fit(x_train, y_train)

    def predict(self, x: array) -> array:
        """
        Returns the predicted real-valued targets for all samples

        Args:
            x: (N,D) array with D-dimensional samples

        Returns: (N,) array
        """
        # Call sklearn predict method
        return self._model.predict(x)
