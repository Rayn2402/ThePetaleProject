"""
Author: Nicolas Raymond

This file is used wrap the sklearn random forest models within PetaleClassifier and PetaleRegressor
abstract classes
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.models.wrappers.sklearn_wrappers import SklearnBinaryClassifierWrapper, SklearnRegressorWrapper
from src.utils.hyperparameters import CategoricalHP, NumericalContinuousHP, NumericalIntHP
from typing import Optional


class PetaleBinaryRFC(SklearnBinaryClassifierWrapper):
    """
    Sklearn random forest classifier wrapper
    """
    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2, max_features: str = "auto",
                 max_depth: int = 10, max_samples: float = 1, classification_threshold: int = 0.5,
                 weight: Optional[float] = None):
        """
        Creates a sklearn random forest classifier model and sets required protected attributes using
        parent's constructor

        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_depth: the maximum depth of the three
            max_samples: percentage of samples drawn from X to train each base estimator
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        super().__init__(model=RandomForestClassifier(n_estimators=n_estimators,
                                                      min_samples_split=min_samples_split,
                                                      max_features=max_features,
                                                      max_samples=max_samples,
                                                      max_depth=max_depth,
                                                      criterion="entropy"),
                         classification_threshold=classification_threshold, weight=weight)

    @staticmethod
    def get_hps():
        return list(RandomForestHP()) + [RandomForestHP.WEIGHT]


class PetaleRFR(SklearnRegressorWrapper):
    """
    Sklearn random forest regressor wrapper
    """

    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2,
                 max_features: str = "auto", max_samples: float = 1, max_depth: int = 10):
        """
        Creates a sklearn random forest regression model and sets protected attributes using parent's constructor
        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_samples: percentage of samples drawn from X to train each base estimator
            max_depth: the maximum depth of the three
        """
        super().__init__(model=RandomForestRegressor(n_estimators=n_estimators,
                                                     min_samples_split=min_samples_split,
                                                     max_features=max_features,
                                                     max_samples=max_samples,
                                                     max_depth=max_depth))

    @staticmethod
    def get_hps():
        return list(RandomForestHP())


class RandomForestHP:
    """
    Random forest's hyperparameters
    """
    MAX_DEPTH = NumericalIntHP("max_depth")
    MAX_FEATURES = CategoricalHP("max_features")
    MAX_SAMPLES = NumericalContinuousHP("max_samples")
    MIN_SAMPLES_SPLIT = NumericalIntHP("min_samples_split")
    N_ESTIMATORS = NumericalIntHP("n_estimators")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.MAX_DEPTH, self.MAX_FEATURES, self.MAX_SAMPLES,
                     self.MIN_SAMPLES_SPLIT, self.N_ESTIMATORS])
