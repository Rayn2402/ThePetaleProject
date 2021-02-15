"""
Author : Nicolas Raymond

This file contains the sampling function used to separate test set from train set
"""
import numpy as np
import pandas as pd


def stratified_sample(df, target_col, n, quantiles=4):
    """
    Proceeds to a stratified sampling of the original dataset

    :param df: pandas dataframe
    :param target_col: name of the column to use
    :param n: sample size, if 0 < n < 1 we consider n as a percentage of data to select
    :param quantiles: number of quantiles to used if the target_col is continuous
    :return: pandas dataframe
    """
    if target_col not in df.columns:
        raise Exception('Target column not part of the dataframe')
    if n < 0:
        raise Exception('n must be greater than 0')

    # If n is a percentage we change it to a integer
    elif 0 < n < 1:
        n = int(n*df.shape[0])

    if len(df[target_col].unique()) > 10:
        df["quantiles"] = pd.qcut(df[target_col], quantiles, labels=False)
        target_col = "quantiles"

    sample = df.groupby(target_col, group_keys=False).\
        apply(lambda x: x.sample(int(np.rint(n*len(x)/len(df))))).sample(frac=1).reset_index(drop=True)

    sample = sample.drop(['quantiles'], axis=1, errors='ignore')
    df = df.drop(['quantiles'], axis=1, errors='ignore')

    return sample
