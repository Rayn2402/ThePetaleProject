"""

Filename: sklearn_wrappers.py

Author: Nicolas Raymond

Description: This file is used to define the abstract classes
             used as wrappers for models with the sklearn API

Date of last modification : 2022/04/13
"""
import os
import pickle

from numpy import array
from src.data.processing.datasets import PetaleDataset
from src.models.abstract_models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.utils.hyperparameters import HP
from typing import Any, Callable, List, Dict, Optional


class SklearnBinaryClassifierWrapper(PetaleBinaryClassifier):
    """
    Class used as a wrapper for binary classifier with sklearn API
    """
    def __init__(self,
                 model_params: Dict[str, Any],
                 classification_threshold: float = 0.5,
                 weight: Optional[float] = None,
                 train_params: Optional[Dict[str, Any]] = None):

        """
        Sets the model protected attributes and other protected attributes via parent's constructor

        Args:
            model_params: parameters used to initialize the classification model with sklearn API
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
            train_params: training parameters proper to model for fit function
        """
        # Initialization of model
        self._model_params = model_params
        self._model = None

        # Call of parent's constructor
        super().__init__(classification_threshold=classification_threshold,
                         weight=weight,
                         train_params=train_params)

    def _update_pos_scaling_factor(self, y_train: array) -> None:
        """
        Updates the scaling factor that needs to be apply to samples in class 1

        Args:
            y_train: y_train: (N, 1) array with labels

        Returns: None
        """
        raise NotImplementedError

    def fit(self, dataset: PetaleDataset) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) array with D-dimensional samples
                     - y : (N,) array with classification labels
                     - idx : (N,) array with idx of samples according to the whole dataset

        Returns: None
        """
        # We extract train set
        x_train, y_train, _ = dataset[dataset.train_mask]

        # We update the positive scaling factor
        self._update_pos_scaling_factor(y_train=y_train)

        # Call the fit method
        self._model.fit(x_train, y_train, **self.train_params)

    def predict_proba(self,
                      dataset: PetaleDataset,
                      mask: Optional[List[int]] = None) -> array:
        """
        Returns the probabilities of being in class 1 for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) array with D-dimensional samples
                     - y : (N,) array with classification labels
                     - idx : (N,) array with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to predict proba

        Returns: (N,) array
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        # Call predict_proba method and takes the prediction for class 1
        proba = self._model.predict_proba(x)[:, 1]

        return proba.squeeze()

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """
        # We save the model with pickle
        filepath = os.path.join(path, "sklearn_model.sav")
        pickle.dump(self._model, open(filepath, "wb"))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        raise NotImplementedError


class SklearnRegressorWrapper(PetaleRegressor):
    """
    Class used as a wrapper for regression model with sklearn API
    """
    def __init__(self,
                 model: Callable,
                 train_params: Optional[Dict[str, Any]] = None):
        """
        Sets the model protected attribute and the training params using parent's constructor

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
            dataset: PetaleDatasets which its items are tuples (x, y) where
                     - x : (N,D) array with D-dimensional samples
                     - y : (N,) array with classification labels
                     - idx : (N,) array with idx of samples according to the whole dataset

        Returns: None
        """
        # We extract train set
        x_train, y_train, _ = dataset[dataset.train_mask]

        # Call the fit method
        self._model.fit(x_train, y_train, **self.train_params)

    def predict(self,
                dataset: PetaleDataset,
                mask: Optional[List[int]] = None) -> array:
        """
        Returns the predicted real-valued targets for all samples in
        a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) array with D-dimensional samples
                     - y : (N,) array with classification labels
                     - idx : (N,) array with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to make predictions

        Returns: (N,) array
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the data
        x, _, _ = dataset[mask]

        # Call sklearn predict method
        return self._model.predict(x)

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """
        # We save the model with pickle
        filepath = os.path.join(path, "sklearn_model.sav")
        pickle.dump(self._model, open(filepath, "wb"))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        raise NotImplementedError
