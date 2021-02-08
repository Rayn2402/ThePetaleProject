"""
Authors : Nicolas Raymond

This files contains all class function related to preprocessing

"""
from .Transforms import ContinuousTransform as ConT
from .Transforms import CategoricalTransform as CaT


def preprocess_continuous(df):
    """
    Applies all continuous transforms to a dataframe containing only continuous data

    :param df: pandas dataframe
    :return: pandas dataframe
    """
    return ConT.normalize(ConT.fill_missing(ConT.to_float(df)))


def preprocess_categoricals(df):
    """
    Applies all categorical transforms to a dataframe containing only continuous data
    :param df: pandas dataframe
    :return: pandas dataframe, list of encoding sizes
    """
    return CaT.one_hot_encode(CaT.to_category(df))