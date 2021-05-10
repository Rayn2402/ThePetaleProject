"""
Authors : Nicolas Raymond

Files that contains class related to Datasets

"""
from typing import Optional, List, Callable, Sequence, Tuple
from pandas import DataFrame, Series
from torch.utils.data import Dataset
from torch import from_numpy, cat, ones, tensor
from Data.Preprocessing import preprocess_continuous, preprocess_categoricals
from Data.Transforms import ContinuousTransform as ConT
from SQL.constants import *


class PetaleDataset(Dataset):

    def __init__(self, df: DataFrame, cont_cols: List[str], target: str,
                 cat_cols: Optional[List[str]] = None, split: bool = True,
                 add_biases: bool = False, mean: Optional[Series] = None,
                 std: Optional[Series] = None, mode: Optional[Series] = None):
        """
        Creates a petale dataset where categoricals columns are separated from the continuous by default.

        :param df: pandas dataframe
        :param cont_cols: list with names of continuous columns
        :param target: string with target column name
        :param cat_cols: list with names of categorical columns
        :param split: boolean indicating if categorical features must remain separated from continuous features
        :param add_biases: boolean indicating if a column of ones should be added at the beginning of X_cont
        :param mean: means to use for data normalization (pandas series)
        :param std : stds to use for data normalization (pandas series)
        :param mode: modes to use in order to fill missing categorical values (pandas series)
        """
        assert PARTICIPANT in df.columns, 'IDs missing from the dataframe'

        # We save the survivors ID
        self.IDs = df[PARTICIPANT]

        # We save and preprocess continuous features
        self.X_cont = preprocess_continuous(df[cont_cols], mean, std)
        self.X_cont = from_numpy(self.X_cont.values).float()

        # We save the number of elements in the datasets
        self.N = self.IDs.shape[0]

        # We add biases to continuous features if required
        if add_biases:
            self.X_cont = cat((ones(self.N, 1), self.X_cont), 1)

        # We preprocess and save categorical features if there are some
        if cat_cols is not None:
            if split:
                # We keep ordinal encodings of categorical features separated from continuous features
                self.X_cat = preprocess_categoricals(df[cat_cols], mode=mode)
                self.X_cat = from_numpy(self.X_cat.values).float()
            else:
                # We concatenate one-hot encodings of categorical features with continuous features
                self.X_cat = preprocess_categoricals(df[cat_cols], encoding='one-hot', mode=mode)
                self.X_cat = from_numpy(self.X_cat.values).float()
                self.__concat_dataset()
        else:
            self.X_cat = None

        # We save the targets
        self.y = from_numpy(ConT.to_float(df[target]).values).float().flatten()

        # We define the getter function according to the presence of absence of categorical features
        self.getter = self.define_getter(cat_cols, split)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: Sequence[int]) -> Tuple[tensor, tensor, Optional[tensor]]:
        return self.getter(idx)

    def define_getter(self, cat_cols: List[str], split: bool) -> Callable:
        """
        Builds to right __getitem__ function according to cat_cols value

        :param cat_cols: list of categorical columns names
        :param split: boolean indicating if categorical features must remain separated from continuous features
        :return: function
        """
        if (cat_cols is not None) and split:
            def f(idx):
                return self.X_cont[idx, :], self.X_cat[idx, :], self.y[idx]

        else:
            def f(idx):
                return self.X_cont[idx, :], self.y[idx]

        return f

    def __concat_dataset(self) -> None:
        """
        Concatenates categorical and continuous data
        WARNING! : !Categorical and continuous must remain separated if we want to have an embedding layer!
        """
        self.X_cont = cat((self.X_cont, self.X_cat), 1)
        self.X_cat = None


class PetaleDataframe:

    def __init__(self, df: DataFrame, cont_cols: List[str], target: str,
                 cat_cols: Optional[List[str]] = None, mean: Optional[Series] = None,
                 std: Optional[Series] = None, mode: Optional[Series] = None, **kwargs):
        """
        Applies transformations to a dataframe and store the result as the dataset for the Random Forest model

        :param df: pandas dataframe
        :param cont_cols: list with names of continuous columns
        :param target: string with target column name
        :param cat_cols: list with names of categorical columns
        :param mean: means to use for data normalization (pandas series)
        :param std : stds to use for data normalization (pandas series)
        :param mode: modes to use in order to fill missing categorical values (pandas series)
        """

        assert PARTICIPANT in df.columns, 'IDs missing from the dataframe'

        # We save the survivors ID
        self.IDs = df[PARTICIPANT]

        # We save the number of elements in the datasets
        self.N = self.IDs.shape[0]

        # We save and preprocess continuous features
        self.X_cont = df[cat_cols + cont_cols].copy()
        self.X_cont[cont_cols] = preprocess_continuous(self.X_cont[cont_cols], mean, std)

        # We save and preprocess categorical features
        if cat_cols is not None:
            self.X_cont[cat_cols] = preprocess_categoricals(self.X_cont[cat_cols], mode=mode)

        # We save the targets
        self.y = ConT.to_float(df[target]).values.flatten()

        # We set the categorical data to none
        self.X_cat = None

    def __len__(self) -> int:
        return self.N
