"""
Author : Nicolas Raymond

This file contains all transformations related to preprocessing treatment
"""

from typing import Optional
import pandas as pd


class ContinuousTransform:
    """
    Class of transformation that can be applied to continuous data
    """
    @staticmethod
    def to_float(df: pd.DataFrame) -> pd.DataFrame:
        """
        Changes type of pandas columns to float
        """
        return df.astype('float')

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


class CategoricalTransform:
    """
    Class of transformation that can be applied to categorical data
    """

    @staticmethod
    def to_category(df: pd.DataFrame) -> pd.DataFrame:
        """
        Changes type of pandas column to category
        """
        return df.astype('category')

    @staticmethod
    def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
        """
        One hot encodes all columns of the dataframe
        """
        return pd.get_dummies(df)

    @staticmethod
    def ordinal_encode(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies ordinal encoding to all columns of the dataframe
        """
        for c in df.columns:
            df[c] = df[c].cat.codes

        return df

    @staticmethod
    def fill_missing(df: pd.DataFrame, mode: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fills missing values of continuous data columns with mode
        """
        if mode is not None:
            return df.fillna(mode)
        else:
            return df.fillna(df.mode().iloc[0])

