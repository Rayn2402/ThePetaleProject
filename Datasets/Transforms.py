"""
Author : Nicolas Raymond

This file contains all transformations related to preprocessing treatment
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class ContinuousTransform:
    """
    Class of transformation that can be applied to continuous data
    """
    @staticmethod
    def to_float(df):
        """
        Changes type of pandas columns to float
        """
        return df.astype('float')

    @staticmethod
    def normalize(df):
        """
        Applies normalization to columns of a pandas dataframe
        """
        return (df-df.mean())/df.std()

    @staticmethod
    def fill_missing(df):
        """
        Fills missing value of continuous data columns with mean
        """
        return df.fillna(df.mean())


class CategoricalTransform:
    """
    Class of transformation that can be applied to categorical data
    """

    @staticmethod
    def to_category(df):
        """
        Changes type of pandas column to category
        """
        return df.astype('category')

    @staticmethod
    def one_hot_encode(df):
        """
        One hot encodes all columns of the dataframe
        """
        return pd.get_dummies(df)

    @staticmethod
    def ordinal_encode(df):
        """
        Applies ordinal encoding to all columns of the dataframe
        """
        for c in df.columns:
            df[c] = df[c].cat.codes

        return df

