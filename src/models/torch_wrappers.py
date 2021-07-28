"""
Author: Nicolas Raymond

This file store the classification and regression wrapper classes for torch custom model
"""

from src.models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.data.processing.datasets import PetaleDataset
from torch import tensor
from typing import Any, Callable, Dict, Optional


class TorchBinaryClassifierWrapper(PetaleBinaryClassifier):
    """
    Class used as a wrapper for binary classifier with sklearn API
    """
    def __init__(self, model: Callable, classification_threshold: float = 0.5,
                 weight: Optional[float] = None, train_params: Optional[Dict[str, Any]] = None):

        """
        Sets model protected attribute and other protected attributes via parent's constructor

        Args:
            model: classification model with sklearn API
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
            train_params: training parameters proper to model for fit function
        """
        self._model = model
        super().__init__(classification_threshold=classification_threshold, weight=weight,
                         train_params=train_params)

    def fit(self, dataset: PetaleDataset) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: None
        """
        # We extract train set
        _, y_train, _ = dataset[list(range(len(dataset)))]

        # We get sample weights
        sample_weights = self.get_sample_weights(y_train)

        # Call the fit method
        self._model.fit(dataset, sample_weights=sample_weights, **self.train_params)

    def predict_proba(self, dataset: PetaleDataset) -> tensor:
        """
        Returns the probabilities of being in class 1 for all samples in the test set

        Args:
            dataset: PetaleDatasets which items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: (N,) array
        """
        # Call predict_proba method, takes the prediction for class 1 and squeeze the array
        proba = self._model.predict_proba(dataset)

        return proba.squeeze()


class TorchRegressorWrapper(PetaleRegressor):
    """
    Class used as a wrapper for regression model with sklearn API
    """
    def __init__(self, model: Callable, train_params: Optional[Dict[str, Any]] = None):
        """
        Sets the model protected attribute

        Args:
            model: regression model with sklearn API
            train_params: training parameters proper to model for fit function
        """
        self._model = model
        super().__init__(train_params=train_params)

    def fit(self, dataset: PetaleDataset) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which items are tuples (x, y) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels

        Returns: None
        """
        # Call the fit method
        self._model.fit(dataset, **self.train_params)

    def predict(self, dataset: PetaleDataset) -> tensor:
        """
        Returns the predicted real-valued targets for all samples in the test set

        Args:
            dataset: PetaleDatasets which items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset

        Returns: (N,) array
        """

        # Call sklearn predict method
        return self._model.predict(dataset)
