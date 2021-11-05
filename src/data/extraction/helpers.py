import os
import csv

from pathlib import Path
from pandas import DataFrame
from settings.paths import Paths
from src.data.extraction.constants import *
from typing import Optional

KEY = "CHROM_POS"
SECONDS_IN_YEAR = 31556952


def reformat_string(table_name):
    """
    Changes a string to an appropriate format to use as filename or directory

    :param table_name: string
    :return: string
    """
    return table_name.replace(".", "").replace(": ", "").replace("?", "").replace("/", "")


def timeDeltaToYears(timeDelta):
    """
    Function that transforms from the type TimeDelta to years

    :return: number of years
    """

    return round(timeDelta.total_seconds() / SECONDS_IN_YEAR, 2)


def AbsTimeLapse(df, new_col, first_date, second_date):
    """
    Computes a new column that gives the absolute differences (in years) between two column dates

    :param df: pandas dataframe
    :param new_col: new column name (for the column that will store the results)
    :param first_date: first date column name
    :param second_date: second date column name
    """
    df[new_col] = abs(df[second_date] - df[first_date])
    df[new_col] = df[new_col].apply(timeDeltaToYears)


def extract_var_id(var_name):
    """
    Function that returns the id of the variable of a given variable

    :param var_name: the variable name
    :return:a string
    """

    return var_name.split()[0]


def check_categorical_var(data):
    """
    Function that gets the data of a variable and return True if this variable is categorical

    :param data:the data of the variable
    :return: Bool
    """
    for item in data:
        if item is not None:
            if isinstance(item, str):
                return True

    if len(data.unique()) > 10:
        return False
    return True


def retrieve_categorical(df, ids):
    """
    Function that return a dataframe containing only categorical variables from a given dataframe

    :param df: a pandas dataframe
    :return:a string
    """
    categorical_cols = [
        col for col in df.columns if (check_categorical_var(df[col]))]
    for col_id in ids:
        if col_id not in categorical_cols:
            categorical_cols.append(col_id)
    return df[categorical_cols]


def retrieve_numerical(df, ids):
    """
    Function that return a dataframe containing only numerical variables from a given dataframe

    :param df: a pandas dataframe
    :return:a string
    """
    numerical_cols = [
        col for col in df.columns if (not check_categorical_var(df[col]))]
    for col_id in ids:
        if col_id not in numerical_cols:
            numerical_cols.append(col_id)
    return df[numerical_cols]


def get_column_stats(df, col):
    """
    Retrieves statistic from a numerical column in a pandas dataframe

    :param df: pandas dataframe
    :param col: name of the columne
    :return: mean, var, max, min
    """
    numerical_data = df[col].astype("float")
    mean = round(numerical_data.mean(axis=0), 2)
    std = round(numerical_data.std(axis=0), 2)
    min = numerical_data.min()
    max = numerical_data.max()

    return mean, std, min, max


def fill_id(id):
    """
    Add characters missing to ID
    :param id: current id (string)
    """
    return f"P" + "".join(["0"]*(3-len(id))) + id


def get_missing_update(df):
    """
    Prints the number of rows and the number of missing values for each column
    :param df: pandas dataframe
    """
    print("Current number of rows : ", df.shape[0])
    print("Missing counts : ")
    print(df.isnull().sum(axis=0), "\n\n")


def pivot_snp_dataframe(df: DataFrame, snps_id_filter: Optional[DataFrame] = None) -> DataFrame:
    """
    Filter the patients snps table and execute a transposition

    :param df: pandas dataframe
    :param snps_id_filter: list with snps id to keep
    :return: pandas dataframe
    """
    df[KEY] = df[CHROM].astype(str) + "_" + df[SNPS_POSITION].astype(str)

    # We filter the table to only keep rows where CHROM and POS match with an SNP in the top 5
    if snps_id_filter is not None:
        df = df[df[KEY].isin(list(snps_id_filter[KEY].values))]

    # We dump CHROM and POS columns
    df = df.drop([CHROM, SNPS_POSITION, REF, ALT, GENE_REF_GEN], axis=1, errors='ignore')

    # We change index for CHROM_POS column
    df = df.set_index(KEY)

    # We transpose the dataframe
    df = df.T
    df.index.rename(PARTICIPANT, inplace=True)
    df.reset_index(inplace=True)

    return df

