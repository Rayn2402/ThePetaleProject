"""
Author: Nicolas Raymond

This file is used wrap the sklearn random forest models within PetaleClassifier and PetaleRegressor
abstract classes
"""
from numpy import array, zeros, where
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.models.base_models import PetaleBinaryClassifier, PetaleRegressor
from typing import Optional, Tuple


class PetaleBinaryRFC(PetaleBinaryClassifier):
    """
    Sklearn random forest classifier wrapper
    """
    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2, max_features: str = "auto",
                 max_samples: float = 1, classification_threshold: int = 0.5,
                 weight: Optional[float] = None):
        """
        Sets required protected attributes and creates a sklearn random forest classifier model
        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_samples: percentage of samples drawn from X to train each base estimator
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        # We call parent's constructor
        super().__init__(classification_threshold=classification_threshold, weight=weight)

        # We build the model
        self.__model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                              max_features=max_features, max_samples=max_samples)

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
            sample_weights[where(y_train == 0)] = w0
            sample_weights[where(y_train == 1)] = w1
        else:
            sample_weights = None

        # Call the sklearn fit method
        self.__model.fit(x_train, y_train, sample_weights)

    def predict_proba(self, x: array) -> array:
        """
        Returns the probabilities of being in class 1 for all samples

        Args:
            x: (N,D) array with D-dimensional samples

        Returns: (N,) array
        """
        # Call sklearn predict_proba method, takes the prediction for class 1 and squeeze the array
        proba = self.__model.predict_proba(x)[:, 1]

        return proba.squeeze()


class PetaleRFR(PetaleRegressor):
    """
    Sklearn random forest regressor wrapper
    """
    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2,
                 max_features: str = "auto", max_samples: float = 1):
        """
        Sets required protected attributes and creates a sklearn random forest classifier model
        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_samples: percentage of samples drawn from X to train each base estimator
        """
        self.__model = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                             max_features=max_features, max_samples=max_samples)

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
        self.__model.fit(x_train, y_train)

    def predict(self, x: array) -> array:
        """
        Returns the predicted real-valued targets for all samples

        Args:
            x: (N,D) array with D-dimensional samples

        Returns: (N,) array
        """
        # Call sklearn predict method
        return self.__model.predict(x)
