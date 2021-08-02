"""
Author: Nicolas Raymond

This file is used wrap the sklearn random forest models within PetaleClassifier and PetaleRegressor
abstract classes
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.models.wrappers.sklearn_wrappers import SklearnBinaryClassifierWrapper, SklearnRegressorWrapper
from typing import Optional


class PetaleBinaryRFC(SklearnBinaryClassifierWrapper):
    """
    Sklearn random forest classifier wrapper
    """
    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2, max_features: str = "auto",
                 max_samples: float = 1, classification_threshold: int = 0.5,
                 weight: Optional[float] = None):
        """
        Creates a sklearn random forest classifier model and sets required protected attributes using
        parent's constructor

        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_samples: percentage of samples drawn from X to train each base estimator
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        super().__init__(model=RandomForestClassifier(n_estimators=n_estimators,
                                                      min_samples_split=min_samples_split,
                                                      max_features=max_features,
                                                      max_samples=max_samples,
                                                      criterion="entropy"),
                         classification_threshold=classification_threshold, weight=weight)


class PetaleRFR(SklearnRegressorWrapper):
    """
    Sklearn random forest regressor wrapper
    """
    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2,
                 max_features: str = "auto", max_samples: float = 1):
        """
        Creates a sklearn random forest regression model and sets protected attributes using parent's constructor
        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_samples: percentage of samples drawn from X to train each base estimator
        """
        super().__init__(model=RandomForestRegressor(n_estimators=n_estimators,
                                                     min_samples_split=min_samples_split,
                                                     max_features=max_features,
                                                     max_samples=max_samples))
