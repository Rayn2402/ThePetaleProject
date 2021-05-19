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
from typing import Optional, List, Callable, Tuple, Union, Any


class CustomDataset(ABC):
    """
    Scaffolding of all dataset classes implemented for our experiments
    """
    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None):
        """
        Sets private and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data

        """
        assert PARTICIPANT in df.columns, "Patients' ids missing from the dataframe."
        assert (cont_cols is not None or cat_cols is not None), "At least a list of continuous columns" \
                                                                " or a list of categorical columns must be given."
        for columns in [cont_cols, cat_cols]:
            self.check_columns_validity(df, columns)

        # We call super init since we're using ABC
        super().__init__()

        # Set private attributes
        self._ids = df[PARTICIPANT]
        self._target = target
        self._train_mask, self._valid_mask, self._test_mask = [], [], []
        self._original_data = df
        self._n = df.shape[0]
        self._x_cont, self._x_cat = None, None
        self._y = self._initialize_targets(target)

        # Set public attributes
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols

        # We set two "getter" method to get modes and encodings of categorical columns
        self._get_modes, self._get_encodings = self._define_categorical_stats_getter(cat_cols)

        # We set a "getter" method to get mu ans std of continuous columns
        self._get_mu_and_std = self._define_numerical_stats_getter(cont_cols)

        # We set two "setter" methods to update available data after masks update
        self._set_numerical = self._define_numerical_data_setter(cont_cols)
        self._set_categorical = self._define_categorical_data_setter(cat_cols)

        # We update current training mask with all the data
        self._update_masks(list(range(self._n)), [], [])

    def __len__(self) -> int:
        return self._n

    def _current_train_stats(self) -> Tuple[Optional[Series], Optional[Series], Optional[Series], Optional[dict]]:
        """
        Returns the current statistics and encodings related to the training data
        """
        # We extract the current training data
        train_data = self._original_data.iloc[self._train_mask]

        # We compute the current values of mu, std, modes and encodings
        mu, std = self._get_mu_and_std(train_data)
        modes = self._get_modes(train_data)
        enc = self._get_encodings(train_data)

        return mu, std, modes, enc

    def _define_categorical_data_setter(self, cat_cols: Optional[List[str]] = None) -> Callable:
        """
        Defines the function used to set categorical data after masks update
        """
        if cat_cols is None:
            def set_categorical(modes: Optional[Series], enc: Optional[dict]) -> None:
                pass

            return set_categorical

        else:
            return self._categorical_setter

    def _define_categorical_stats_getter(self, cat_cols: Optional[List[str]] = None) -> Tuple[Callable, Callable]:
        """
        Defines the function used to extract the modes of categorical columns and defines the function
        to extract encodings of the categorical columns in a dataframe
        """
        if cat_cols is None:
            def get_modes(df: Optional[DataFrame]) -> None:
                return None

            def get_encodings(df: Optional[DataFrame]) -> None:
                return None
        else:
            # Make sure that categorical data in the original dataframe is in the correct format
            self._original_data[cat_cols] = self._original_data[cat_cols].astype('category')

            def get_modes(df: DataFrame) -> Series:
                return df[cat_cols].mode().iloc[0]

            def get_encodings(df: DataFrame) -> dict:
                return {c: {v: k for k, v in enumerate(df[c].cat.categories)} for c in cat_cols}

        return get_modes, get_encodings

    def _define_numerical_data_setter(self, cont_cols: Optional[List[str]] = None) -> Callable:
        """
        Defines the function used to set numerical continuous data after masks update
        """
        if cont_cols is None:
            def set_numerical(mu: Optional[Series], std: Optional[Series]) -> None:
                pass

            return set_numerical
        else:
            return self._numerical_setter

    def _define_numerical_stats_getter(self, cont_cols: Optional[List[str]] = None) -> Callable:
        """
        Defines the function used to extract the mean and the standard deviations of numerical columns
        in a dataframe.
        """
        if cont_cols is None:
            def get_mu_and_std(df: DataFrame) -> Tuple[None, None]:
                return None, None
        else:
            # Make sure that numerical data in the original dataframe is in the correct format
            self._original_data[cont_cols] = self._original_data[cont_cols].astype(float)

            def get_mu_and_std(df: DataFrame) -> Tuple[Series, Series]:
                return df[self.cont_cols].mean(), df[self.cont_cols].std()

        return get_mu_and_std

    def _update_masks(self, train_mask: List[int], test_mask: List[int],
                      valid_mask: Optional[List[int]] = None) -> None:
        """
        Updates the train, valid and test masks and then preprocess the data available
        according to the current statistics of the training data
        """
        # We set the new masks values
        self._train_mask, self._valid_mask, self._test_mask = train_mask, valid_mask, test_mask

        # We compute the current values of mu, std, modes and encodings
        mu, std, modes, enc = self._current_train_stats()

        # We update the data that will be available via __get_item__
        self._set_numerical(mu, std)
        self._set_categorical(modes, enc)

    @abstractmethod
    def _numerical_setter(self, mu: Series, std: Series) -> None:
        """
        Fills missing values of numerical continuous data according according to the means of the
        training set and then normalize continuous data using the means and the standard
        deviations of the training set.
        """
        raise NotImplementedError

    @abstractmethod
    def _categorical_setter(self, modes: Series, enc: dict) -> None:
        """
        Fill missing values of categorical data according to the modes in the training set and
        then encodes categories using the same encoding as in the training set.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_targets(self, target: str) -> Union[tensor, array]:
        """
        Returns the targets container in the appropriate format (tensor or array)
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


class PetaleDataset(CustomDataset, Dataset):
    """
    Datasets used to train, valid and tests our neural network models
    """
    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None):
        """
        Sets private and public attributes of the class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
        """
        # We use the _init_ of the parent class CustomDataset
        CustomDataset.__init__(self, df, target, cont_cols, cat_cols)

        # We define the item getter function
        self._item_getter = self._define_item_getter(cont_cols, cat_cols)

    def __len__(self) -> int:
        return CustomDataset.__len__(self)

    def __getitem__(self, idx: Any) -> Tuple[tensor, tensor, tensor]:
        return self._item_getter(idx)

    def _categorical_setter(self, modes: Series, enc: dict) -> None:
        """
        Fill missing values of categorical data according to the modes in the training set and
        then encodes categories using the same ordinal encoding as in the training set.

        Args:
            modes: modes of the categorical column according to the training mask
            enc: ordinal encodings used for categorical columns according to the training mask

        Returns: None
        """
        # We apply an ordinal encoding to categorical columns
        temporary_df, _ = preprocess_categoricals(self._original_data[self.cat_cols].copy(), mode=modes, encodings=enc)
        self._x_cat = from_numpy(temporary_df.values).float()

    def _define_item_getter(self, cont_cols: Optional[List[str]] = None,
                            cat_cols: Optional[List[str]] = None) -> Callable:
        """
        Defines the function that must be used by __get_item__ in order to get an item data
        Args:
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data

        Returns: item_getter function
        """
        if cont_cols is None:
            def item_getter(idx: Any) -> Tuple[None, tensor, tensor]:
                return None, self._x_cat[idx, :], self._y[idx]

        elif cat_cols is None:
            def item_getter(idx: Any) -> Tuple[tensor, None, tensor]:
                return self._x_cont[idx, :], None, self._y[idx]

        else:
            def item_getter(idx: Any) -> Tuple[tensor, tensor, tensor]:
                return self._x_cont[idx, :], self._x_cat[idx, :], self._y[idx]

        return item_getter

    def _initialize_targets(self, target: str) -> Union[tensor, array]:
        """
        Saves targets values in a tensor

        Args:
            target: name of the column with the targets

        Returns: tensor
        """
        temporary_df = self._original_data[target].astype(float)
        return from_numpy(temporary_df.values).float().flatten()

    def _numerical_setter(self, mu: Series, std: Series) -> None:
        """
        Fills missing values of numerical continuous data according according to the means of the
        training set and then normalize continuous data using the means and the standard
        deviations of the training set.

        Args:
            mu: means of the numerical column according to the training mask
            std: standard deviations of the numerical column according to the training mask

        Returns: None
        """
        # We fill missing with means and normalize the data
        temporary_df = preprocess_continuous(self._original_data[self.cont_cols].copy(), mu, std)
        self._x_cont = from_numpy(temporary_df.values).float()


#
# class PetaleDataframe:
#
#     def _init_(self, df: DataFrame, cont_cols: List[str], target: str,
#                  cat_cols: Optional[List[str]] = None, mean: Optional[Series] = None,
#                  std: Optional[Series] = None, mode: Optional[Series] = None,
#                  encodings: Optional[dict] = None, **kwargs):
#         """
#         Applies transformations to a dataframe and store the result as the dataset for the Random Forest model
#
#         :param df: pandas dataframe
#         :param cont_cols: list with names of continuous columns
#         :param target: string with target column name
#         :param cat_cols: list with names of categorical columns
#         :param mean: means to use for data normalization (pandas series)
#         :param std : stds to use for data normalization (pandas series)
#         :param mode: modes to use in order to fill missing categorical values (pandas series)
#         :param encodings: dict of dict with integers to use as encoding for each category's values
#         """
#
#         assert PARTICIPANT in df.columns, 'IDs missing from the dataframe'
#
#         # We save the survivors ID
#         self.IDs = df[PARTICIPANT]
#
#         # We save the number of elements in the datasets
#         self.N = self.IDs.shape[0]
#
#         # We save and preprocess continuous features
#         self.X_cont = df[cat_cols + cont_cols].copy() if cat_cols is not None else df[cont_cols].copy()
#         self.X_cont[cont_cols] = preprocess_continuous(self.X_cont[cont_cols], mean, std)
#
#         # We save and preprocess categorical features
#         if cat_cols is not None:
#             self.X_cont[cat_cols], self.encodings = preprocess_categoricals(self.X_cont[cat_cols], mode=mode,
#                                                                             encodings=encodings)
#         # We save the targets
#         self.y = ConT.to_float(df[target]).values.flatten()
#
#         # We set the categorical data to none
#         self.X_cat = None
#
#     def _len_(self) -> int:
#         return self.N
