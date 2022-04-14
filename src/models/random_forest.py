"""
Filename: random_forest.py

Authors: Nicolas Raymond

Description: This file is used to define the regression and classification
             wrappers for the sklearn random forest models

Date of last modification : 2022/04/13
"""

from numpy import array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.models.wrappers.sklearn_wrappers import SklearnBinaryClassifierWrapper, SklearnRegressorWrapper
from src.utils.hyperparameters import CategoricalHP, HP, NumericalContinuousHP, NumericalIntHP
from typing import List, Optional


class PetaleBinaryRFC(SklearnBinaryClassifierWrapper):
    """
    Sklearn random forest classifier wrapper for the Petale framework
    """
    def __init__(self,
                 n_estimators: int = 100,
                 min_samples_split: int = 2,
                 max_features: str = "auto",
                 max_leaf_nodes: int = 100,
                 max_samples: float = 1,
                 classification_threshold: int = 0.5,
                 weight: Optional[float] = None):
        """
        Creates a sklearn random forest classifier model and sets other protected
        attributes using parent's constructor

        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_leaf_nodes: the maximum number of leaf nodes
            max_samples: percentage of samples drawn from X to train each base estimator
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        super().__init__(model_params=dict(n_estimators=n_estimators,
                                           min_samples_split=min_samples_split,
                                           max_features=max_features,
                                           max_samples=max_samples,
                                           max_leaf_nodes=max_leaf_nodes,
                                           criterion="entropy"),
                         classification_threshold=classification_threshold,
                         weight=weight)

    def _update_pos_scaling_factor(self, y_train: array) -> None:
        """
        Calculates the positive scaling factor and creates a new model

        Args:
            y_train: (N, 1) array with labels

        Returns: None
        """
        self._model = RandomForestClassifier(**self._model_params,
                                             class_weight={0: 1, 1: self._get_scaling_factor(y_train)})

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(RandomForestHP()) + [RandomForestHP.WEIGHT]


class PetaleRFR(SklearnRegressorWrapper):
    """
    Sklearn random forest regressor wrapper for the Petale framework
    """

    def __init__(self,
                 n_estimators: int = 100,
                 min_samples_split: int = 2,
                 max_features: str = "auto",
                 max_samples: float = 1,
                 max_leaf_nodes: int = 100):
        """
        Creates a sklearn random forest regression model and sets other protected
        attributes using parent's constructor

        Args:
            n_estimators: number of trees in the forest
            min_samples_split: minimum number of samples required to split an internal node
            max_features: number of features to consider when looking for the best split {“auto”, “sqrt”, “log2”}
            max_samples: percentage of samples drawn from X to train each base estimator
            max_leaf_nodes: the maximum number of leaf nodes
        """
        super().__init__(model=RandomForestRegressor(n_estimators=n_estimators,
                                                     min_samples_split=min_samples_split,
                                                     max_features=max_features,
                                                     max_samples=max_samples,
                                                     max_leaf_nodes=max_leaf_nodes))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(RandomForestHP())


class RandomForestHP:
    """
    Random forest's hyperparameters
    """
    MAX_LEAF_NODES = NumericalIntHP("max_leaf_nodes")
    MAX_FEATURES = CategoricalHP("max_features")
    MAX_SAMPLES = NumericalContinuousHP("max_samples")
    MIN_SAMPLES_SPLIT = NumericalIntHP("min_samples_split")
    N_ESTIMATORS = NumericalIntHP("n_estimators")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.MAX_LEAF_NODES, self.MAX_FEATURES, self.MAX_SAMPLES,
                     self.MIN_SAMPLES_SPLIT, self.N_ESTIMATORS])
