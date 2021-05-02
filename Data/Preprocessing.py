"""
Authors : Nicolas Raymond

This files contains all class function related to preprocessing

"""
from Data.Transforms import ContinuousTransform as ConT
from Data.Transforms import CategoricalTransform as CaT
from SQL.NewTablesScripts.constants import *
from Data.outlier_analysis import get_outlier_ids

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
    assert encoding in ENCODING, 'Encoding option not available'

    # We ensure that all columns are considered as categories
    df = CaT.to_category(df)

    if encoding == "ordinal":
        return CaT.ordinal_encode(df)

    else:
        return CaT.one_hot_encode(df)


def preprocess_outliers(df, cont_cols, cat_cols=None, ids=None):
    """
    Function that deletes the outliers from agiven dataframe

    :param df: the dataframe containing the data
    :param cont_cols: the continuous columns in this dataframe
    :param cat_cols: the categorical columns in this dataframe
    :param ids: list of the outliers ids
    """

    # We prepare the ids of the rows to be removed
    if ids is None:
        print("hello")
        ids = get_outlier_ids(df=df, cont_cols=cont_cols, cat_cols=cat_cols)

    # We return the dataframe after deleting the outliers
    return df[~df[PARTICIPANT].isin(ids)]
