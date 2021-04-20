"""
Author : Nicolas Raymond

This file contains the Sampler class used to separate test sets from train sets
"""

from typing import Sequence, Union, Tuple
from SQL.DataManager.Utils import PetaleDataManager
from Data.Datasets import PetaleDataset, PetaleDataframe
from Data.Transforms import ContinuousTransform as ConT
from SQL.NewTablesScripts.constants import *
import numpy as np
import pandas as pd


class Sampler:

    def __init__(self, dm: PetaleDataManager, table_name: str, cont_cols: Sequence[str],
                 target_col: str, cat_cols: Union[None, Sequence[str]] = None, to_dataset: bool = True):
        """
        Object that creates all datasets
        :param dm: PetaleDataManager
        :param table_name: name of the table on which we want to sample datasets
        :param cont_cols: list with the names of continuous columns of the table
        :param cat_cols: list with the names of the categorical columns of the table
        :param target_col: name of the target column in the table
        :param to_dataset: bool indicating if we want a PetaleDataset (True) or a PetaleDataframe (False)
        """

        # We save the learning set as seen in the Workflow presentation
        self.learning_set = dm.get_table(table_name)

        # We make sure that continuous variables are considered as continuous
        self.learning_set[cont_cols] = ConT.to_float(self.learning_set[cont_cols])

        # We save the continuous and categorical column names
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols

        # We save the target column name
        self.target_col = target_col

        # We save the dataset class constructor
        self.dataset_constructor = PetaleDataset if to_dataset else PetaleDataframe

    def __call__(self, k: int = 10, l: int = 1, split_cat: bool = True,
                 valid_size: Union[int, float] = 0.20, test_size: Union[int, float] = 0.20,
                 add_biases: bool = False) -> dict:

        return self.create_train_and_test_datasets(k, l, split_cat, valid_size, test_size, add_biases)

    def create_train_and_test_datasets(self, k: int = 10, l: int = 1, split_cat: bool = True,
                                       valid_size: Union[int, float] = 0.20, test_size: Union[int, float] = 0.20,
                                       add_biases: bool = False) -> dict:
        """
        Creates the train and test PetaleDatasets from the df and the specified continuous and categorical columns

        :param k: number of outer validation loops
        :param l: number if inner validation loops
        :param split_cat: boolean indicating if we want to split categorical variables from the continuous ones
        :param valid_size: number of elements in the valid set (if 0 < valid < 1 we consider the parameter as a %)
        :param test_size: number of elements in the test set (if 0 < test_size < 1 we consider the parameter as a %)
        :param add_biases: boolean indicating if a column of ones should be added at the beginning of X_cont
        :return: dictionary with all datasets
        """

        # We initialize an empty dictionary to store the outer loops datasets
        all_datasets = {}

        seeds = [19, 2021, 26, 2, 1999, 1010, 54, 777, 3059, 631]

        # We create the datasets for the outer validation loops:
        for i in range(k):

            # We split the training and test data
            train, test = split_train_test(self.learning_set, self.target_col, test_size, random_state=seeds[i])
            train, valid = split_train_test(train, self.target_col, valid_size, random_state=seeds[i])
            outer_dict = self.dataframes_to_datasets(train, valid, test, split_cat, add_biases)

            # We add storage in the outer dict to save the inner loops datasets
            outer_dict['inner'] = {}

            # We create the datasets for the inner validation loops
            for j in range(l):

                in_train, in_test = split_train_test(train, self.target_col, test_size)
                in_train, in_valid = split_train_test(in_train, self.target_col, valid_size)
                outer_dict['inner'][j] = self.dataframes_to_datasets(in_train, in_valid, in_test, split_cat, add_biases)
                all_datasets[i] = outer_dict

        return all_datasets

    def dataframes_to_datasets(self, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                               split_cat: bool = True, add_biases: bool = False) -> dict:
        """
        Turns three pandas dataframe into training, valid and test PetaleDatasets

        :param train: Pandas dataframe with training data
        :param valid: Pandas dataframe with valid data
        :param test: Pandas dataframe with test data
        :param split_cat: boolean indicating if we want to split categorical variables from the continuous ones
        :param add_biases: boolean indicating if a column of ones should be added at the beginning of X_cont
        :return: dict
        """
        # We save the mean and the standard deviations of the continuous columns in train
        mean, std = train[self.cont_cols].mean(), train[self.cont_cols].std()

        # We create the train, valid and test datasets
        train_ds = self.dataset_constructor(df=train, cont_cols=self.cont_cols, target=self.target_col,
                                            cat_cols=self.cat_cols, split=split_cat, add_biases=add_biases)

        if valid is not None:
            valid_ds = self.dataset_constructor(df=valid, cont_cols=self.cont_cols, target=self.target_col,
                                                cat_cols=self.cat_cols, split=split_cat, mean=mean, std=std,
                                                add_biases=add_biases)
        else:
            valid_ds = None

        test_ds = self.dataset_constructor(df=test, cont_cols=self.cont_cols, target=self.target_col,
                                           cat_cols=self.cat_cols, split=split_cat, mean=mean, std=std,
                                           add_biases=add_biases)

        return {"train": train_ds, "valid": valid_ds, "test": test_ds}

    @staticmethod
    def visualize_splits(datasets: dict) -> None:
        """
        Details the data splits for the experiment

        :param datasets: dict with all datasets obtain from the Sampler
        """
        print("#----------------------------------#")
        for k, v in datasets.items():
            print(f"Split {k+1} \n")
            print(f"Outer :")
            valid = v['valid'] if v['valid'] is not None else []
            print(f"Train {len(v['train'])} - Valid {len(valid)} - Test {len(v['test'])}")
            print("Inner")
            for k1, v1 in v['inner'].items():
                valid = v1['valid'] if v1['valid'] is not None else []
                print(f"{k+1}.{k1} -> Train {len(v1['train'])} - Valid {len(valid)} -"
                      f" Test {len(v1['test'])}")
            print("#----------------------------------#")


def get_warmup_sampler(dm: PetaleDataManager, to_dataset: bool = True):
    """
    Creates a Sampler for the WarmUp data table
    :param dm: PetaleDataManager
    :param to_dataset: bool indicating if we want a PetaleDataset (True) or a PetaleDataframe (False)
    """
    cont_cols = [WEIGHT, TDM6_HR_END, TDM6_DIST, DT, AGE, MVLPA]
    return Sampler(dm, LEARNING_0, cont_cols, VO2R_MAX, to_dataset=to_dataset)


def get_learning_one_sampler(dm: PetaleDataManager, to_dataset: bool = True):
    """
    Creates a Sampler for the Learning One data table
    :param dm: PetaleDataManager
    :param to_dataset: bool indicating if we want a PetaleDataset (True) or a PetaleDataframe (False)
    """
    # We save continuous columns
    cont_cols = [AGE, HEIGHT, WEIGHT, AGE_AT_DIAGNOSIS, DT, TSEOT, RADIOTHERAPY_DOSE, TDM6_DIST, TDM6_HR_END,
                 TDM6_HR_REST, TDM6_TAS_END, TDM6_TAD_END, MVLPA, TAS_REST, TAD_REST, DOX]

    # We save the categorical columns
    cat_cols = [SEX, SMOKING, DEX_PRESENCE]

    return Sampler(dm, LEARNING_1, cont_cols, FITNESS_LVL, cat_cols, to_dataset)


def split_train_test(df: pd.DataFrame, target_col: str,
                     test_size: Union[int, float] = 0.20,
                     random_state: bool = None) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """
    Split de training and testing data contained within a pandas dataframe
    :param df: pandas dataframe
    :param target_col: name of the target column
    :param test_size: number of elements in the test set (if 0 < test_size < 1 we consider the parameter as a %)
    :param random_state: seed for random number generator (does not overwrite global seed value)
    :return: 2 pandas dataframe
    """

    if test_size > 0:

        # Test and train split
        test_data = stratified_sample(df, target_col, test_size, random_state=random_state)
        train_data = df.drop(test_data.index)
        return train_data, test_data

    else:
        return df, None


def stratified_sample(df: pd.DataFrame, target_col: str, n: Union[int, float],
                      quantiles: int = 4, random_state: Union[int, None] = None) -> pd.DataFrame:
    """
    Proceeds to a stratified sampling of the original dataset based on the target variable

    :param df: pandas dataframe
    :param target_col: name of the column to use for stratified sampling
    :param n: sample size, if 0 < n < 1 we consider n as a percentage of data to select
    :param quantiles: number of quantiles to used if the target_col is continuous
    :param random_state: seed for random number generator (does not overwrite global seed value)
    :return: pandas dataframe
    """
    assert target_col in df.columns, 'Target column not part of the dataframe'
    assert n > 0, 'n must be greater than 0'

    # If n is a percentage we change it to a integer
    if 0 < n < 1:
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
        apply(lambda x: x.sample(int(np.rint(n*len(x)/len(sample))), random_state=random_state)).\
        sample(frac=1, random_state=random_state)

    sample = sample.drop(['quantiles'], axis=1, errors='ignore')

    return sample


