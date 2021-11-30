"""
Filename: helpers.py

Author: Nicolas Raymond
        Mehdi Mitiche

Description: This file contains helpful functions for pandas dataframe manipulations

Date of last modification : 2021/11/05
"""

from datetime import timedelta
from pandas import DataFrame, Series
from typing import List, Optional, Tuple

SECONDS_IN_YEAR = 31556925


def convert_timedelta_to_years(time_delta: timedelta) -> float:
    """
    Converts a timedelta to a number of years

    Args:
        time_delta: timedelta object

    Returns: number of years
    """
    return round(time_delta.total_seconds() / SECONDS_IN_YEAR, 2)


def get_abs_years_timelapse(df: DataFrame,
                            new_col: str,
                            first_date: str,
                            second_date: str) -> None:
    """
    Computes a new column that gives the absolute differences (in years)
    between two dates columns

    Args:
        df: pandas dataframe
        new_col: new column name (for the column that will store the results)
        first_date: first date column name
        second_date: second date column name

    Returns: None
    """

    df[new_col] = abs(df[second_date] - df[first_date])
    df[new_col] = df[new_col].apply(convert_timedelta_to_years)


def get_column_stats(df: DataFrame,
                     col: str) -> Tuple[float, float, float, float]:
    """
    Retrieves statistic from a numerical column in a pandas dataframe

    Args:
        df: pandas dataframe
        col: name of the column

    Returns: mean, std, max, min
    """
    numerical_data = df[col].astype("float")
    mean = round(numerical_data.mean(axis=0), 2)
    std = round(numerical_data.std(axis=0), 2)
    min_ = numerical_data.min()
    max_ = numerical_data.max()

    return mean, std, min_, max_


def is_categorical(data: Series) -> bool:
    """
    Verifies if a variable is categorical using its data

    Args:
        data: pandas series (column of a pandas dataframe)

    Returns: True if categorical
    """
    for item in data:
        if isinstance(item, str):
            return True

    if len(data.unique()) > 10:
        return False

    return True


def retrieve_categorical_var(df: DataFrame,
                             to_keep: Optional[List[str]] = None) -> DataFrame:
    """
    Returns a dataframe containing only categorical variables of a given dataframe

    Args:
        df: pandas dataframe
        to_keep: list of columns to keep in the returned dataframe no matter their types

    Returns: pandas dataframe
    """
    # We convert the "to_keep" parameter into list if was not provided
    if to_keep is None:
        to_keep = []

    # We identify the columns to check
    cols_to_check = [col for col in df.columns if col not in to_keep]

    # We identify the categorical columns
    categorical_cols = [col for col in cols_to_check if is_categorical(df[col])]

    return df[categorical_cols + to_keep]


def retrieve_numerical_var(df: DataFrame,
                           to_keep: Optional[List[str]] = None) -> DataFrame:
    """
    Returns a dataframe containing only numerical variables of a given dataframe

    Args:
        df: pandas dataframe
        to_keep: list of columns to keep in the returned dataframe no matter their types

    Returns: pandas dataframe
    """
    # We convert the "to_keep" parameter into list if was not provided
    if to_keep is None:
        to_keep = []

    # We identify the columns to check
    cols_to_check = [col for col in df.columns if col not in to_keep]

    # We identify the categorical columns
    categorical_cols = [col for col in cols_to_check if not is_categorical(df[col])]

    return df[categorical_cols + to_keep]


def get_missing_update(df: DataFrame) -> None:
    """
    Prints the number of rows and the number of missing values for each column

    Args:
        df: pandas dataframe

    Returns: None
    """
    print("Current number of rows : ", df.shape[0])
    print("Missing counts : ")
    print(df.isnull().sum(axis=0), "\n\n")
