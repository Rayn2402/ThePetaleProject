"""
Author: Nicolas Raymond

This file is used wrap the sklearn random forest models within PetaleClassifier and PetaleRegressor
abstract classes
"""
from numpy import array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.models.base_models import PetaleBinaryClassifier, PetaleRegressor
from torch import from_numpy, tensor
from typing import List, Optional, Tuple


class PetaleBinaryRFC(PetaleBinaryClassifier):
    """
    Sklearn random forest classifier wrapper
    """
    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2, max_features: str = "auto",
                 max_samples: float = 1, classification_threshold: int = 0.5,
                 class_weights: Optional[List[float]] = None):
        """
        Sets required protected attributes and creates a sklearn random forest classifier model
        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_samples: percentage of samples drawn from X to train each base estimator
            classification_threshold: threshold used to classify a sample in class 1
            class_weights: weights attributes to samples of each class
        """
        # We call parent's constructor
        super().__init__(classification_threshold=classification_threshold, class_weights=class_weights)

        # We build the model
        class_weights = {i: class_weights[i] for i in range(len(class_weights))} if class_weights is not None else None
        self.__model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                              max_features=max_features, max_samples=max_samples,
                                              class_weight=class_weights)

    def fit(self, x_train: array, y_train: array,
            eval_set: Optional[Tuple[array, array]] = None, **kwargs) -> None:

        # Call the sklearn fit method
        self.__model.fit(x_train, y_train)

    def predict_proba(self, x: array) -> array:

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
            eval_set: Optional[Tuple[array, array]] = None, **kwargs) -> None:

        # Call the sklearn fit method
        self.__model.fit(x_train, y_train)

    def predict(self, x: array) -> tensor:

        # Call sklearn predict method
        return from_numpy(self.__model.predict(x))
