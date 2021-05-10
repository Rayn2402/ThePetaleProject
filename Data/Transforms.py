"""
Author : Nicolas Raymond

This file contains all transformations related to preprocessing treatment
"""

import pandas as pd


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
    def normalize(df, mean=None, std=None):
        """
        Applies normalization to columns of a pandas dataframe
        """
        if mean is not None and std is not None:
            return (df-mean)/std
        else:
            return (df-df.mean())/df.std()

    @staticmethod
    def fill_missing(df, mean=None):
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

    @staticmethod
    def fill_missing(df, mode=None):
        """
        Fills missing values of continuous data columns with mode
        """
        if mode is not None:
            return df.fillna(mode)
        else:
            return df.fillna(df.mode().iloc[0])

