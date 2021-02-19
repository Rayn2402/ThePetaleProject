"""
Authors : Nicolas Raymond

This files contains all class function related to preprocessing

"""
from .Transforms import ContinuousTransform as ConT
from .Transforms import CategoricalTransform as CaT

ENCODING = ["ordinal", "one-hot"]


def preprocess_continuous(df, mean=None, std=None):
    """
    Applies all continuous transforms to a dataframe containing only continuous data

    :param df: pandas dataframe
    :param mean: pandas series with mean
    :param std: pandas series with standard deviations
    :return: pandas dataframe
    """
    return ConT.normalize(ConT.fill_missing(df, mean), mean, std)


def preprocess_categoricals(df, encoding="ordinal"):
    """
    Applies all categorical transforms to a dataframe containing only continuous data

    :param df: pandas dataframe
    :param encoding: one option in ("ordinal", "one-hot")
    :return: pandas dataframe, list of encoding sizes
    """
    if encoding not in ENCODING:
        raise Exception('Encoding option not available')

    # We ensure that all columns are considered as categories
    df = CaT.to_category(df)

    if encoding == "ordinal":
        return CaT.ordinal_encode(df)

    else:
        return CaT.one_hot_encode(df)