"""
Author: Nicolas Raymond

This file is used to store abstract class to use as wrappers for models with the sklearn API
"""
from numpy import array
from src.data.processing.datasets import PetaleDataset
from src.models.base_models import PetaleBinaryClassifier, PetaleRegressor
from typing import Callable, Optional


class SklearnBinaryClassifierWrapper(PetaleBinaryClassifier):
    """
    Class used as a wrapper for binary classifier with sklearn API
    """
    def __init__(self, model: Callable, classification_threshold: float = 0.5,
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

    def fit(self, dataset: PetaleDataset, **kwargs) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which items are tuples (x, y) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels

        Returns: None
        """
        # We extract train set
        x_train, y_train = dataset[dataset.train_mask]

        # We get sample weights
        sample_weights = self.get_sample_weights(y_train)

        # Call the sklearn fit method
        self._model.fit(x_train, y_train, sample_weight=sample_weights, **kwargs)

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

    def fit(self, dataset: PetaleDataset, **kwargs) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which items are tuples (x, y) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels

        Returns: None
        """
        # We extract train set
        x_train, y_train = dataset[dataset.train_mask]

        # Call the sklearn fit method
        self._model.fit(x_train, y_train, **kwargs)

    def predict(self, x: array) -> array:
        """
        Returns the predicted real-valued targets for all samples

        Args:
            x: (N,D) array with D-dimensional samples

        Returns: (N,) array
        """
        # Call sklearn predict method
        return self._model.predict(x)
