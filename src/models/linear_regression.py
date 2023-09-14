"""
Filename: linear_regression.py

Authors: Nicolas Raymond

Description: This file is used to define the regression
             wrappers for the sklearn linear regression model

Date of last modification : 2022/04/13
"""

from sklearn.linear_model import LinearRegression
from src.models.wrappers.sklearn_wrappers import SklearnRegressorWrapper
from src.utils.hyperparameters import HP
from typing import List


class PetaleLR(SklearnRegressorWrapper):
    """
    Sklearn linear regression wrapper for the Petale framework
    """

    def __init__(self,
                 fit_intercept: bool = True):
        """
        Creates a sklearn random forest regression model and sets other protected
        attributes using parent's constructor

        Args:
            fit_intercept: Whether to calculate the intercept for this model.
        """
        super().__init__(model=LinearRegression(fit_intercept=fit_intercept))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return []
