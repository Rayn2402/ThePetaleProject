"""
Filename: preprocessing.py

Author: Nicolas Raymond

Description: Defines all functions related to preprocessing

Date of last modification : 2021/11/01
"""

from numpy import linspace, nan, quantile
from pandas import DataFrame, Series
from src.data.processing.transforms import ContinuousTransform as ConT
from src.data.processing.transforms import CategoricalTransform as CaT
from typing import Dict, List, Optional, Tuple

ENCODING = ["ordinal", "one-hot"]


def preprocess_continuous(df: DataFrame,
                          mean: Optional[Series] = None,
                          std: Optional[Series] = None) -> DataFrame:
    """
    Applies all continuous transforms to a dataframe containing only continuous data

    Args:
        df: pandas dataframe
        mean: pandas series with mean
        std: pandas series with standard deviations

    Returns: preprocessed dataframe
    """
    return ConT.normalize(ConT.fill_missing(df, mean), mean, std)


def preprocess_categoricals(df: DataFrame,
                            encoding: str = "ordinal",
                            mode: Optional[Series] = None,
                            encodings: Optional[dict] = None) -> Tuple[DataFrame, Optional[dict]]:
    """
    Applies all categorical transforms to a dataframe containing only continuous data

    Args:
        df: pandas dataframe
        encoding: one option in ("ordinal", "one-hot")
        mode: panda series with modes of columns
        encodings: dict of dict with integers to use as encoding for each category's values

    Returns:pandas dataframe, dict of encodings
    """

    if encoding not in ENCODING:
        raise ValueError("Encoding option not available")

    # We ensure that all columns are considered as categories
    df = CaT.fill_missing(df, mode)

    if encoding == "ordinal":
        df, encodings_dict = CaT.ordinal_encode(df, encodings)
        return df, encodings_dict

    else:
        return CaT.one_hot_encode(df), None


def preprocess_for_apriori(df: DataFrame,
                           cont_cols: Optional[Dict[str, int]] = None,
                           cat_cols: Optional[List[str]] = None) -> List[List[str]]:
    """
    Preprocess a dataframe to work with apriori algorithm

    Args:
        df: pandas dataframe
        cont_cols: dictionary with name of continuous columns as key and number of groups
                   we want to create as value.
        cat_cols: list of categorical columns

    Returns: list of list
    """

    # Take the subset needed from the dataframe
    df = df[list(cont_cols.keys()) + cat_cols]

    # Change categorical variables
    for c in cat_cols:
        df.loc[:, c] = df[c].apply(func=lambda x: f"{c}_{x}")
        df.loc[:, c] = df[c].apply(func=lambda x: x if x.split("_")[-1] != "None" else nan)

    # Change numeric variables
    for col, nb_group in cont_cols.items():
        df = create_groups(df, col, nb_group)

    # Extract list of list from dataframe
    records = df.values.tolist()

    # Remove nans from lists
    for i, _ in enumerate(records):
        records[i] = remove_nan(records[i])

    return records


def create_groups(df: DataFrame,
                  cont_col: str,
                  nb_group: int) -> DataFrame:
    """
    Change each value of the column cont_col for its belonging group
    computed using nb_group quantiles

    Args:
        df: pandas dataframe
        cont_col: name of the continuous column
        nb_group: number of quantiles (group) wanted

    Returns: dataframe with modified column
    """

    if nb_group < 2:
        raise ValueError('Must have at least 2 groups')

    # We compute number needed to calculate quantiles
    percentage = linspace(0, 1, nb_group+1)[1:-1]

    # We sort values in an ascending way
    df = df.sort_values(by=cont_col)

    # We retrieve the numeric value to calculate the quantiles
    data = df.loc[:, cont_col].astype(float).values

    # We set few local variables
    group = cont_col.upper()
    max_ = df[cont_col].max()
    quantiles = []

    # We compute quantiles
    for p in percentage:
        quantiles.append(round(quantile(data, p), 2))

    # We change row values
    j = turn_to_range(df, cont_col, 0, quantiles[0], group=f"{group} <= q1")
    for i in range(1, len(quantiles)):
        j = turn_to_range(df, cont_col, j, quantiles[i], group=f"{group} >q{i-1},<=q{i}")

    _ = turn_to_range(df, cont_col, j, max_+1, group=f"{group} >q{len(quantiles)}")

    return df


def turn_to_range(df: DataFrame,
                  cont_col: str,
                  start_index: int,
                  quantile: float,
                  group: str) -> int:
    """
    Changes categorical values of selected index into a string representing a range

    Args:
        df: pandas dataframe
        cont_col: name of continuous column
        start_index: index where to start the modification
        quantile: quantile to reach
        group: name of the group to be assigned

    Returns: index where the quantile was exceeded
    """

    # We get the index of the column
    j = df.columns.get_loc(cont_col)

    for i in range(start_index, df.shape[0]):
        if df.iloc[i, j] < quantile:
            df.iloc[i, j] = group
        else:
            return i


def remove_nan(record: List[str]) -> List:
    """
    Removes nans from a record

    Args:
        record: list of str

    Returns: curated list
    """
    record = [x for x in record if str(x) != 'nan']

    return record
