"""
Filename: xgboost_.py

Authors: Nicolas Raymond

Description: This file is used to define the regression and classification
             wrappers for the xgboost models

Date of last modification : 2021/10/25
"""

from numpy import array
from src.models.wrappers.sklearn_wrappers import SklearnBinaryClassifierWrapper, SklearnRegressorWrapper
from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP
from typing import List, Optional
from xgboost import XGBClassifier, XGBRegressor


class PetaleBinaryXGBC(SklearnBinaryClassifierWrapper):
    """
    XGBoost classifier wrapper for the Petale framework
    """
    def __init__(self,
                 lr: float = 0.3,
                 max_depth: int = 6,
                 subsample: float = 1,
                 alpha: float = 0,
                 beta: float = 0,
                 classification_threshold: int = 0.5,
                 weight: Optional[float] = None):
        """
        Sets protected attributes using parent's constructor
        Args:
            lr: step size shrinkage used in updates to prevent overfitting. After each boosting step, we can
                directly get the weights of new features, and eta shrinks the feature weights to make the
                boosting process more conservative.
            max_depth: maximum depth of a tree. Increasing this value will make the model more complex
                       and more likely to overfit.
            subsample: subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly
                       sample half of the training data prior to growing trees.
            alpha: L1 regularization term on weights.
            beta: L2 regularization term on weights
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        super().__init__(model_params=dict(learning_rate=lr,
                                           max_depth=max_depth,
                                           subsample=subsample,
                                           reg_alpha=alpha,
                                           reg_lambda=beta,
                                           use_label_encoder=False,
                                           objective="binary:logistic",
                                           eval_metric="logloss"),
                         classification_threshold=classification_threshold,
                         weight=weight)

    def _update_pos_scaling_factor(self, y_train: array) -> None:
        """
        Calculates the positive scaling factor and creates a new model

        Args:
            y_train: (N, 1) array with labels

        Returns: None
        """
        self._model = XGBClassifier(**self._model_params,
                                    scale_pos_weight=self._get_scaling_factor(y_train))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(XGBoostHP()) + [XGBoostHP.WEIGHT]


class PetaleXGBR(SklearnRegressorWrapper):
    """
    XGBoost regressor wrapper for the Petale framework
    """
    def __init__(self,
                 lr: float = 0.3,
                 max_depth: int = 6,
                 subsample: float = 1,
                 alpha: float = 0,
                 beta: float = 0):
        """
        Sets protected attributes using parent's constructor

        Args:
            lr: step size shrinkage used in update to prevents overfitting. After each boosting step, we can
                directly get the weights of new features, and eta shrinks the feature weights to make the
                boosting process more conservative.
            max_depth: maximum depth of a tree. Increasing this value will make the model more complex
                       and more likely to overfit.
            subsample: subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly
                       sample half of the training data prior to growing trees.
            alpha: L1 regularization term on weights.
            beta: L2 regularization term on weights
        """
        super().__init__(model=XGBRegressor(learning_rate=lr,
                                            max_depth=max_depth,
                                            subsample=subsample,
                                            reg_alpha=alpha,
                                            reg_lambda=beta))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(XGBoostHP())


class XGBoostHP:
    """
    XGBoost's hyperparameters
    """
    ALPHA = NumericalContinuousHP("alpha")
    BETA = NumericalContinuousHP("beta")
    LR = NumericalContinuousHP("lr")
    MAX_DEPTH = NumericalIntHP("max_depth")
    SUBSAMPLE = NumericalContinuousHP("subsample")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.ALPHA, self.BETA, self.LR, self.MAX_DEPTH, self.SUBSAMPLE])

