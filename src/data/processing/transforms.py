"""
Author : Nicolas Raymond

This file contains all transformations related to preprocessing treatment
"""

from torch import tensor, from_numpy
from typing import Optional, Tuple
import pandas as pd


class ContinuousTransform:
    """
    Class of transformation that can be applied to continuous data
    """

    @staticmethod
    def normalize(df: pd.DataFrame, mean: Optional[pd.Series] = None,
                  std: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Applies normalization to columns of a pandas dataframe
        """
        if mean is not None and std is not None:
            return (df-mean)/std
        else:
            return (df-df.mean())/df.std()

    @staticmethod
    def fill_missing(df: pd.DataFrame, mean: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fills missing values of continuous data columns with mean
        """
        if mean is not None:
            return df.fillna(mean)
        else:
            return df.fillna(df.mean())

    @staticmethod
    def to_tensor(df: pd.DataFrame) -> tensor:
        """
        Takes a dataframe with categorical columns and return a tensor with "longs"

        Args:
            df: dataframe with categorical columns only

        Returns: tensor
        """
        return from_numpy(df.to_numpy(dtype=float)).long()


class CategoricalTransform:
    """
    Class of transformation that can be applied to categorical data
    """

    @staticmethod
    def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
        """
        One hot encodes all columns of the dataframe
        """
        return pd.get_dummies(df)

    @staticmethod
    def ordinal_encode(df: pd.DataFrame, encodings: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
        """
        Applies ordinal encoding to all columns of the dataframe
        """
        if encodings is None:
            encodings = {}
            for c in df.columns:
                encodings[c] = {v: k for k, v in enumerate(df[c].cat.categories)}
                df[c] = df[c].cat.codes

        else:
            for c in df.columns:
                column_encoding = encodings[c]
                df[c] = df[c].apply(lambda x: column_encoding[x])

        return df, encodings

    @staticmethod
    def fill_missing(df: pd.DataFrame, mode: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fills missing values of continuous data columns with mode
        """
        if mode is not None:
            return df.fillna(mode)
        else:
            return df.fillna(df.mode().iloc[0])

    @staticmethod
    def to_tensor(df: pd.DataFrame) -> tensor:
        """
        Takes a dataframe with numerical columns and return a tensor with "floats"

        Args:
            df: dataframe with categorical columns only

        Returns: tensor
        """
        return from_numpy(df.to_numpy(dtype=float)).float()

