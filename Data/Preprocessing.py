"""
Authors : Nicolas Raymond

This files contains all class function related to preprocessing

"""
from typing import Optional, Tuple
from pandas import DataFrame, Series
from Data.Transforms import ContinuousTransform as ConT
from Data.Transforms import CategoricalTransform as CaT

ENCODING = ["ordinal", "one-hot"]


def preprocess_continuous(df: DataFrame, mean: Optional[Series] = None,
                          std: Optional[Series] = None) -> DataFrame:
    """
    Applies all continuous transforms to a dataframe containing only continuous data

    :param df: pandas dataframe
    :param mean: pandas series with mean
    :param std: pandas series with standard deviations
    :return: pandas dataframe
    """
    return ConT.normalize(ConT.fill_missing(df, mean), mean, std)


def preprocess_categoricals(df: DataFrame, encoding: str = "ordinal",
                            mode: Optional[Series] = None,
                            encodings: Optional[dict] = None) -> Tuple[DataFrame, Optional[dict]]:
    """
    Applies all categorical transforms to a dataframe containing only continuous data

    :param df: pandas dataframe
    :param encoding: one option in ("ordinal", "one-hot")
    :param mode: panda series with modes of columns
    :param encodings: dict of dict with integers to use as encoding for each category's values
    :return: pandas dataframe, list of encoding sizes
    """
    assert encoding in ENCODING, 'Encoding option not available'

    # We ensure that all columns are considered as categories
    df = CaT.fill_missing(df, mode)

    if encoding == "ordinal":
        df, encodings_dict = CaT.ordinal_encode(df, encodings)
        return df, encodings_dict

    else:
        return CaT.one_hot_encode(df), None
