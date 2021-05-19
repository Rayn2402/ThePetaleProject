"""
Authors : Nicolas Raymond

Files that contains class related to Datasets

"""

from abc import ABC, abstractmethod
from Data.Preprocessing import preprocess_continuous, preprocess_categoricals
from Data.Transforms import ContinuousTransform as ConT
from numpy import array
from pandas import DataFrame, Series
from SQL.constants import *
from torch.utils.data import Dataset
from torch import from_numpy, cat, ones, tensor
from typing import Optional, List, Callable, Sequence, Tuple, Union


class CustomDataset(ABC):
    """
    Scaffolding of all dataset classes implemented for our experiments
    """
    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None,
                 bias: bool = False):
        """
        Sets internal private and public attributes of our custom dataset class
        """
        assert PARTICIPANT in df.columns, "Patients' ids missing from the dataframe."
        assert (cont_cols is not None or cat_cols is not None), "At least a list of continuous columns" \
                                                                " or a list of categorical columns must be given."
        for columns in [cont_cols, cat_cols]:
            self.check_columns_validity(df, columns)

        # We call super init since we're using ABC
        super().__init__()

        # Set private attributes
        self.__ids = df[PARTICIPANT]
        self.__target = target
        self.__train_mask, self.__valid_mask, self.__test_mask = [], [], []
        self.__original_data = df
        self.__n = df.shape[0]
        self.__x_cont, self.__x_cat = None, None
        self.__y = self.__initialize_targets(target)

        # Set public attributes
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.bias = bias

        # We set and internal method to get modes and encodings of categorical columns
        self.__get_modes, self.__get_encodings = self.__define_categorical_stats_getter(cat_cols)

        # We set and internal method to get mu ans std of continuous columns
        self.__get_mu_and_std = self.__define_numerical_stats_getter(cont_cols)

        # We update current training mask with all the data
        self.__update_masks(list(range(self.__n)), [], [])

    def __current_train_stats(self) -> Tuple[Optional[Series], Optional[Series], Optional[Series], Optional[dict]]:
        """
        Returns the current statistics and encodings related to the training data
        """
        # We extract the current training data
        train_data = self.__original_data.iloc[self.__train_mask]

        # We compute the current values of mu, std, modes and encodings
        mu, std = self.__get_mu_and_std(train_data)
        modes = self.__get_modes(train_data)
        enc = self.__get_encodings(train_data)

        return mu, std, modes, enc

    def __define_categorical_stats_getter(self, cat_cols: Optional[List[str]] = None) -> Tuple[Callable, Callable]:
        """
        Defines the function to extract the modes of categorical columns and defines the function
        to extract encodings of the categorical columns in a dataframe
        """
        if cat_cols is None:
            def get_modes(df: DataFrame) -> None:
                return None

            def get_encodings(df: DataFrame) -> None:
                return None
        else:
            # Make sure that categorical data in the original dataframe is in the correct format
            self.__original_data[cat_cols] = self.__original_data[cat_cols].astype('category')

            def get_modes(df: DataFrame) -> Series:
                return df[cat_cols].mode().iloc[0]

            def get_encodings(df: DataFrame) -> dict:
                return {c: {v: k for k, v in enumerate(df[c].cat.categories)} for c in cat_cols}

        return get_modes, get_encodings

    def __define_numerical_stats_getter(self, cont_cols: Optional[List[str]] = None) -> Callable:
        """
        Defines the function to extract the mean and the standard deviations of numerical columns
        in a dataframe.
        """
        if cont_cols is None:
            def get_mu_and_std(df: DataFrame) -> Tuple[None, None]:
                return None, None
        else:
            # Make sure that numerical data in the original dataframe is in the correct format
            self.__original_data[cont_cols] = self.__original_data[cont_cols].astype(float)

            def get_mu_and_std(df: DataFrame) -> Tuple[Series, Series]:
                return df[self.cont_cols].mean(), df[self.cont_cols].std()

        return get_mu_and_std

    def __update_masks(self, train_mask: List[int], test_mask: List[int],
                       valid_mask: Optional[List[int]] = None) -> None:

        # We set the new masks values
        self.__train_mask, self.__valid_mask, self.__test_mask = train_mask, valid_mask, test_mask

        # We compute the current values of mu, std, modes and encodings
        mu, std, modes, enc = self.__current_train_stats()

        # We update the preprocessed data
        self.__update_preprocessed_data(mu, std, modes, enc)

    @abstractmethod
    def __initialize_targets(self, target: str) -> Union[tensor, array]:
        """
        Returns the targets container in the appropriate format (tensor or array)
        """
        raise NotImplementedError

    @abstractmethod
    def __update_preprocessed_data(self, mu: Optional[Series], std: Optional[Series],
                                   modes: Optional[Series], enc: Optional[dict]) -> None:
        """
        Preprocess the data according to the current statistics of the training data
        """
        raise NotImplementedError

    @staticmethod
    def check_columns_validity(df: DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Checks if the columns are all in the dataframe
        """
        if columns is not None:
            dataframe_columns = list(df.columns.values)
            for c in columns:
                assert c in dataframe_columns, f"Column {c} is not part of the given dataframe"


class PetaleDataset(Dataset):

    def __init__(self, df: DataFrame, cont_cols: List[str], target: str,
                 cat_cols: Optional[List[str]] = None, split: bool = True,
                 add_biases: bool = False, mean: Optional[Series] = None,
                 std: Optional[Series] = None, mode: Optional[Series] = None,
                 encodings: Optional[dict] = None):
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
        :param encodings: dict of dict with integers to use as encoding for each category's values
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
                self.X_cat, self.encodings = preprocess_categoricals(df[cat_cols], mode=mode, encodings=encodings)
                self.X_cat = from_numpy(self.X_cat.values).float()
            else:
                # We concatenate one-hot encodings of categorical features with continuous features
                self.X_cat, _ = preprocess_categoricals(df[cat_cols], encoding='one-hot', mode=mode)
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
                 std: Optional[Series] = None, mode: Optional[Series] = None,
                 encodings: Optional[dict] = None, **kwargs):
        """
        Applies transformations to a dataframe and store the result as the dataset for the Random Forest model

        :param df: pandas dataframe
        :param cont_cols: list with names of continuous columns
        :param target: string with target column name
        :param cat_cols: list with names of categorical columns
        :param mean: means to use for data normalization (pandas series)
        :param std : stds to use for data normalization (pandas series)
        :param mode: modes to use in order to fill missing categorical values (pandas series)
        :param encodings: dict of dict with integers to use as encoding for each category's values
        """

        assert PARTICIPANT in df.columns, 'IDs missing from the dataframe'

        # We save the survivors ID
        self.IDs = df[PARTICIPANT]

        # We save the number of elements in the datasets
        self.N = self.IDs.shape[0]

        # We save and preprocess continuous features
        self.X_cont = df[cat_cols + cont_cols].copy() if cat_cols is not None else df[cont_cols].copy()
        self.X_cont[cont_cols] = preprocess_continuous(self.X_cont[cont_cols], mean, std)

        # We save and preprocess categorical features
        if cat_cols is not None:
            self.X_cont[cat_cols], self.encodings = preprocess_categoricals(self.X_cont[cat_cols], mode=mode,
                                                                            encodings=encodings)
        # We save the targets
        self.y = ConT.to_float(df[target]).values.flatten()

        # We set the categorical data to none
        self.X_cat = None

    def __len__(self) -> int:
        return self.N
