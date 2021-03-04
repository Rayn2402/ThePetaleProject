"""
Author : Nicolas Raymond

This file contains the sampling function used to separate test set from train set
"""
import numpy as np
import pandas as pd


def stratified_sample(df, target_col, n, quantiles=4):
    """
    Proceeds to a stratified sampling of the original dataset based on the target variable

    :param df: pandas dataframe
    :param target_col: name of the column to use for stratified sampling
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

    # We make a deep copy of the current dataframe
    sample = df.copy()

    # If the column on which we want to do a stratified sampling is continuous,
    # we create another discrete column based on quantiles
    if len(df[target_col].unique()) > 10:
        sample["quantiles"] = pd.qcut(sample[target_col], quantiles, labels=False)
        target_col = "quantiles"

    # We execute the sampling
    sample = sample.groupby(target_col, group_keys=False).\
        apply(lambda x: x.sample(int(np.rint(n*len(x)/len(sample))))).sample(frac=1)

    sample = sample.drop(['quantiles'], axis=1, errors='ignore')

    return sample


def split_train_test(df, target_col, test_size=0.20):
    """
    Split de training and testing data contained within a pandas dataframe
    :param df: pandas dataframe
    :param target_col: name of the target column
    :param test_size: number of elements in the test set (if 0 < test_size < 1 we consider the parameter as a %)
    :return: 2 pandas dataframe
    """
    test_data = stratified_sample(df, target_col, test_size)
    train_data = df.drop(test_data.index)

    return train_data, test_data

