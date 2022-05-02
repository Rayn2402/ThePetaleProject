"""

Filename: torch_wrappers.py

Author: Nicolas Raymond

Description: This file is used to define the abstract classes
             used as wrappers for custom torch models

Date of last modification : 2022/04/13
"""

import os

from src.data.processing.datasets import PetaleDataset
from src.models.abstract_models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.utils.hyperparameters import HP
from torch import save, tensor
from typing import Any, Callable, Dict, List, Optional


class TorchBinaryClassifierWrapper(PetaleBinaryClassifier):
    """
    Class used as a wrapper for binary classifier inheriting from TorchCustomModel
    """
    def __init__(self,
                 model_constructor: Callable,
                 model_params: Dict[str, Any],
                 classification_threshold: float = 0.5,
                 weight: Optional[float] = None,
                 train_params: Optional[Dict[str, Any]] = None):

        """
        Sets the model protected attribute and other protected attributes via parent's constructor

        Args:
            model_constructor: function used to create the classification model inheriting from TorchCustomModel
            model_params: parameters used to initialize the classification model inheriting from TorchCustomModel
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
            train_params: training parameters proper to model for fit function
        """
        # Initialization of model
        self._model_constructor = model_constructor
        self._model_params = model_params
        self._model = None

        super().__init__(classification_threshold=classification_threshold,
                         weight=weight,
                         train_params=train_params)

    @property
    def model(self) -> Callable:
        return self._model

    def _update_pos_scaling_factor(self, y_train: tensor) -> None:
        """
        Updates the scaling factor that needs to be apply to samples in class 1

        Args:
            y_train: (N, 1) tensor with labels

        Returns: None
        """

        self._model = self._model_constructor(**self._model_params,
                                              pos_weight=self._get_scaling_factor(y_train))

    def fit(self, dataset: PetaleDataset) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset

        Returns: None
        """
        # We extract all labels
        _, y_train, _ = dataset[dataset.train_mask]

        # We update the positive scaling factor
        self._update_pos_scaling_factor(y_train)

        # Call the fit method
        self._model.fit(dataset, **self.train_params)

    def plot_evaluations(self, save_path: Optional[str] = None) -> None:
        """
        Plots the training and valid curves saved

        Args:
            save_path: path where the figures will be saved

        Returns: None
        """
        self._model.plot_evaluations(save_path=save_path)

    def predict_proba(self,
                      dataset: PetaleDataset,
                      mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the probabilities of being in class 1 for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to predict proba

        Returns: (N,) tensor
        """
        # Call predict_proba method, takes the prediction for class 1 and squeeze the array
        proba = self._model.predict_proba(dataset, mask)

        return proba.squeeze()

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """

        save(self._model.state_dict(), os.path.join(path, "torch_model.pt"))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        raise NotImplementedError


class TorchRegressorWrapper(PetaleRegressor):
    """
    Class used as a wrapper for regression model inheriting from TorchCustomModel
    """
    def __init__(self,
                 model: Callable,
                 train_params: Optional[Dict[str, Any]] = None):
        """
        Sets the model protected attribute and train params via parent's constructor

        Args:
            model: regression model inheriting from TorchCustomModel
            train_params: training parameters proper to model for fit function
        """
        self._model = model
        super().__init__(train_params=train_params)

    @property
    def model(self) -> Callable:
        return self._model

    def fit(self, dataset: PetaleDataset) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset

        Returns: None
        """
        # Call the fit method
        self._model.fit(dataset, **self.train_params)

    def plot_evaluations(self, save_path: Optional[str] = None) -> None:
        """
        Plots the training and valid curves saved

        Args:
            save_path: path where the figures will be saved

        Returns: None
        """
        self._model.plot_evaluations(save_path=save_path)

    def predict(self, dataset: PetaleDataset, mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the predicted real-valued targets for all samples in
        a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: List of dataset idx for which we want to make predictions

        Returns: (N,) tensor
        """

        # Call the predict method
        return self._model.predict(dataset, mask)

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path

        Args:
            path: save path

        Returns: None
        """
        save(self._model.state_dict(), os.path.join(path, "torch_model.pt"))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        raise NotImplementedError
