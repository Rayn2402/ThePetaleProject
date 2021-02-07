"""
Authors : Nicolas Raymond

This files contains all class function related to preprocessing

"""
import pandas as pd


class ContinuousTransform:
    """
    Class of transformation that can be applied to continuous data
    """
    type = "cont"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class CategoricalTransform:
    """
    Class of transformation that can be applied to categorical data
    """
    type = "cat"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ToFloat(ContinuousTransform):
    """
    Changes type of pandas columns to float
    """
    def __call__(self, df):
        return df.astype('float')


class Normalize(ContinuousTransform):
    """
    Applies normalization to columns of a pandas dataframe
    """
    def __call__(self, df):
        return (df-df.mean())/df.std()


class FillMissing(ContinuousTransform):
    """
    Fills missing value of continuous data columns with mean
    """
    def __call__(self, df):
        return df.fillna(df.mean())


class ToCategory(CategoricalTransform):
    """
    Changes type of pandas column to category
    """
    def __call__(self, df):
        return df.astype('category')


class OneHotEncode(CategoricalTransform):
    """
    Replaces category levels with new binary columns
    """
    def __call__(self, df):
        encoding_sizes = [len(df[col].cat.categories) for col in df.columns]
        return pd.get_dummies(df), encoding_sizes

