"""
Authors : Nicolas Raymond

Files that contains class related to Datasets

"""

from abc import ABC, abstractmethod
from dgl import heterograph
from numpy import array, concatenate
from pandas import DataFrame, Series
from sklearn.preprocessing import PolynomialFeatures
from src.data.extraction.constants import *
from src.data.processing.preprocessing import preprocess_continuous, preprocess_categoricals
from src.data.processing.transforms import CategoricalTransform as CaT
from src.data.processing.transforms import ContinuousTransform as ConT
from torch.utils.data import Dataset
from torch import from_numpy, tensor, empty
from typing import Optional, List, Callable, Tuple, Union, Any, Dict


class CustomDataset(ABC):
    """
    Scaffolding of all dataset classes implemented for our experiments
    """
    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None,
                 classification: bool = True, target_to_tensor: bool = False):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            classification: True for classification task, False for regression
            target_to_tensor: True if we want the targets to be in a tensor, False for numpy array

        """
        assert PARTICIPANT in df.columns, "Patients' ids missing from the dataframe."
        assert (cont_cols is not None or cat_cols is not None), "At least a list of continuous columns" \
                                                                " or a list of categorical columns must be given."
        for columns in [cont_cols, cat_cols]:
            self._check_columns_validity(df, columns)

        # We call super init since we're using ABC
        super().__init__()

        # Set protected attributes
        self._ids = list(df[PARTICIPANT].values)
        self._target = target
        self._train_mask, self._valid_mask, self._test_mask = [], None, []
        self._original_data = df
        self._n = df.shape[0]
        self._x = df.drop([PARTICIPANT, target], axis=1).copy()
        self._x_cat, self._x_cont = None, None
        self._y = self._initialize_targets(df[target], classification, target_to_tensor)

        # Set public attributes
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols

        # We set a "getter" method to get modes categorical columns and also extract encodings
        self._get_modes, self._encodings = self._define_categorical_stats_getter(cat_cols)

        # We set a "getter" method to get mu ans std of continuous columns
        self._get_mu_and_std = self._define_numerical_stats_getter(cont_cols)

        # We set two "setter" methods to update available data after masks update
        self._set_numerical = self._define_numerical_data_setter(cont_cols)
        self._set_categorical = self._define_categorical_data_setter(cat_cols)

        # We update current training mask with all the data
        self.update_masks(list(range(self._n)), [], [])

    def __len__(self) -> int:
        return self._n

    @property
    def encodings(self) -> Dict[str, Dict[str, int]]:
        return self._encodings

    @property
    def ids(self) -> List[str]:
        return self._ids

    @property
    def original_data(self) -> DataFrame:
        return self._original_data

    @property
    def test_mask(self) -> List[int]:
        return self._test_mask

    @property
    def train_mask(self) -> List[int]:
        return self._train_mask

    @property
    def valid_mask(self) -> Optional[List[int]]:
        return self._valid_mask

    @property
    def x(self) -> DataFrame:
        return self._x

    @property
    def x_cat(self) -> Optional[Union[array, tensor]]:
        return self._x_cat

    @property
    def x_cont(self) -> Optional[Union[array, tensor]]:
        return self._x_cont

    @property
    def y(self) -> array:
        return self._y

    def _categorical_setter(self, modes: Series) -> None:
        """
        Fill missing values of categorical data according to the modes in the training set and
        then encodes categories using the same ordinal encoding as in the training set.

        Args:
            modes: modes of the categorical column according to the training mask

        Returns: None
        """
        # We apply an ordinal encoding to categorical columns
        self._x[self.cat_cols], _ = preprocess_categoricals(self._original_data[self.cat_cols].copy(),
                                                            mode=modes, encodings=self._encodings)
        # We update x_cat protected attribute
        self._set_x_cat()

    def _define_categorical_data_setter(self, cat_cols: Optional[List[str]] = None) -> Callable:
        """
        Defines the function used to set categorical data after masks update
        """
        if cat_cols is None:
            def set_categorical(modes: Optional[Series]) -> None:
                pass

            return set_categorical

        else:
            return self._categorical_setter

    def _define_categorical_stats_getter(self, cat_cols: Optional[List[str]] = None
                                         ) -> Tuple[Callable, Dict[str, Dict[str, int]]]:
        """
        Defines the function used to extract the modes of categorical columns
        """
        if cat_cols is None:
            def get_modes(df: Optional[DataFrame]) -> None:
                return None

            encodings = None
        else:
            # Make sure that categorical data in the original dataframe is in the correct format
            self._original_data[cat_cols] = self._original_data[cat_cols].astype('category')

            # We extract ordinal encodings
            encodings = {c: {v: k for k, v in enumerate(self._original_data[c].cat.categories)} for c in cat_cols}

            def get_modes(df: DataFrame) -> Series:
                return df[cat_cols].mode().iloc[0]

        return get_modes, encodings

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
        self._x[self.cont_cols] = preprocess_continuous(self._original_data[self.cont_cols].copy(), mu, std)

        # We update x_cont protected attribute
        self._set_x_cont()

    def _set_x_cat(self) -> None:
        """
        Abstract method that takes imputed categorical data and convert it
        to proper format for training

        Returns: None
        """
        pass

    def _set_x_cont(self) -> None:
        """
        Abstract method that takes imputed continuous data and convert it
        to proper format for training

        Returns: None
        """
        pass

    def current_train_stats(self) -> Tuple[Optional[Series], Optional[Series], Optional[Series]]:
        """
        Returns the current statistics and encodings related to the training data
        """
        # We extract the current training data
        train_data = self._original_data.iloc[self._train_mask]

        # We compute the current values of mu, std, modes and encodings
        mu, std = self._get_mu_and_std(train_data)
        modes = self._get_modes(train_data)

        return mu, std, modes

    def update_masks(self, train_mask: List[int], test_mask: List[int],
                     valid_mask: Optional[List[int]] = None) -> None:
        """
        Updates the train, valid and test masks and then preprocess the data available
        according to the current statistics of the training data
        """
        # We set the new masks values
        self._train_mask, self._valid_mask, self._test_mask = train_mask, valid_mask, test_mask

        # We compute the current values of mu, std, modes and encodings
        mu, std, modes = self.current_train_stats()

        # We update the data that will be available via __get_item__
        self._set_numerical(mu, std)
        self._set_categorical(modes)

    @abstractmethod
    def __getitem__(self, idx: Union[int, List[int]]) -> Any:
        raise NotImplementedError

    # @abstractmethod
    # def _create_subset(self) -> Any:
    #     """
    #     Returns a subset of the current dataset while using feature selection
    #
    #     Returns: instance of the same class
    #     """
    #     raise NotImplementedError

    @staticmethod
    def _initialize_targets(targets_column: Series, classification: bool,
                            target_to_tensor: bool) -> Union[array, tensor]:
        """
        Sets the targets according to the task and the choice of container
        Args:
            targets_column: column of the dataframe with the targets
            classification: True for classification task, False for regression
            target_to_tensor: True if we want the targets to be in a tensor, False for numpy array

        Returns: targets
        """
        # Set targets protected attribute according to task
        t = targets_column.to_numpy(dtype=float)
        if (not classification) and target_to_tensor:
            t = from_numpy(t).float()
        elif classification:
            if target_to_tensor:
                t = from_numpy(t).long()
            else:
                t = t.astype(int)

        return t.squeeze()

    @staticmethod
    def _check_columns_validity(df: DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Checks if the columns are all in the dataframe
        """
        if columns is not None:
            dataframe_columns = list(df.columns.values)
            for c in columns:
                assert c in dataframe_columns, f"Column {c} is not part of the given dataframe"


class PetaleNNDataset(CustomDataset, Dataset):
    """
    Dataset used to train, valid and tests our neural network models
    """
    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None,
                 classification: bool = True):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            classification: True for classification task, False for regression

        """
        # We use the _init_ of the parent class CustomDataset
        CustomDataset.__init__(self, df, target, cont_cols, cat_cols, classification,
                               target_to_tensor=True)

        # We define the item getter function
        self._item_getter = self._define_item_getter(cont_cols, cat_cols)

    def __len__(self) -> int:
        return CustomDataset.__len__(self)

    def __getitem__(self, idx: Any) -> Tuple[tensor, tensor, tensor]:
        return self._item_getter(idx)

    def _define_item_getter(self, cont_cols: Optional[List[str]] = None,
                            cat_cols: Optional[List[str]] = None) -> Callable:
        """
        Defines the function that must be used by __get_item__ in order to get data
        Args:
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data

        Returns: item_getter function
        """
        if cont_cols is None:
            def item_getter(idx: Any) -> Tuple[tensor, tensor, tensor]:
                return empty(0), self.x_cat[idx, :], self.y[idx]

        elif cat_cols is None:
            def item_getter(idx: Any) -> Tuple[tensor, tensor, tensor]:
                return self.x_cont[idx, :], empty(0), self.y[idx]

        else:
            def item_getter(idx: Any) -> Tuple[tensor, tensor, tensor]:
                return self.x_cont[idx, :], self.x_cat[idx, :], self.y[idx]

        return item_getter

    def _set_x_cat(self) -> None:
        """
        Sets x_cat protected attribute after masks update
        Returns: None
        """
        self._x_cat = CaT.to_tensor(self.x[self.cat_cols])

    def _set_x_cont(self) -> None:
        """
        Sets x_cont protected attribute after masks update
        Returns: None
        """
        self._x_cont = ConT.to_tensor(self.x[self.cont_cols])


class PetaleRFDataset(CustomDataset):
    """
    Dataset used to train, valid and tests our random forest models
    """
    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None,
                 classification: bool = True):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            classification: True for classification task, False for regression

        """
        # We use the _init_ of the parent class
        super().__init__(df, target, cont_cols, cat_cols, classification)

    def __getitem__(self, idx) -> Tuple[Series, array]:
        return self.x.iloc[idx], self.y[idx]


class PetaleLinearModelDataset(CustomDataset):
    """
    Dataset used to train and test our linear models from sklearn
    """

    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None,
                 classification: bool = True,  polynomial_degree: int = 1, include_bias: bool = True):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            classification: True for classification task, False for regression
            polynomial_degree: degree of polynomial basis function to apply to the data
            include_bias: True if we want to include bias to the original data
        """
        # We set the protected attributes
        self._basis_function = PolynomialFeatures(degree=polynomial_degree, include_bias=include_bias)

        # We use the _init_ of the parent class
        super().__init__(df, target, cont_cols, cat_cols, classification)

        # We define the item getter function
        self._item_getter = self._define_item_getter(cont_cols, cat_cols)

    def __getitem__(self, idx) -> Tuple[array, array]:
        return self._item_getter(idx)

    def _define_item_getter(self, cont_cols: Optional[List[str]] = None,
                            cat_cols: Optional[List[str]] = None) -> Callable:
        """
        Defines the function that must be used by __get_item__ in order to get data
        Args:
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data

        Returns: item_getter function
        """
        if cont_cols is None:
            def item_getter(idx: Any) -> Tuple[array, array]:
                return self.x_cat[idx, :], self.y[idx]

        elif cat_cols is None:
            def item_getter(idx: Any) -> Tuple[array, array]:
                return self.x_cont[idx, :], self.y[idx]

        else:
            def item_getter(idx: Any) -> Tuple[array, array]:
                return concatenate((self.x_cont[idx, :], self.x_cat[idx, :]), axis=1), self.y[idx]

        return item_getter

    def _set_x_cat(self) -> None:
        """
        Sets x_cat protected attribute after masks update
        Returns: None
        """
        self._x_cat = self.x[self.cat_cols].to_numpy(dtype=int)

    def _set_x_cont(self) -> None:
        """
        Sets x_cont protected attribute after masks update
        Returns: None
        """
        self._x_cont = self._basis_function.fit_transform(self.x[self.cont_cols].to_numpy(dtype=float))


class PetaleGNNDataset(PetaleNNDataset):
    """
    Dataset used to train, valid and test our Graph Neural Network
    """

    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None,
                 classification: bool = True):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            classification: True for classification task, False for regression

        """
        # We initialize protected attributes proper to GNNDataset class
        self._graph = None

        # We use the _init_ of the parent class CustomDataset
        super().__init__(df, target, cont_cols, cat_cols, classification)

    def _update_graph(self) -> None:
        """
        Updates the graph structures and its data once the masks are updated

        Returns: None
        """
        # We look through categorical columns to generate graph structure
        graph_structure = {}
        for e_types, e_values in self.encodings.items():
            edges_start, edges_end = [], []
            for value in e_values.values():
                idx_subset = self.x.loc[self._x[e_types] == value].index.to_numpy()
                subset_size = idx_subset.shape[0]
                for i in range(subset_size):
                    edges_start += [idx_subset[i]]*(subset_size - 1)
                    remaining_idx = list(range(i)) + list(range(i+1, subset_size))
                    edges_end += list(idx_subset[remaining_idx])
                graph_structure[(PARTICIPANT, e_types, PARTICIPANT)] = (tensor(edges_start), tensor(edges_end))

        # We update the internal graph attribute
        self._graph = heterograph(graph_structure)

        # We set the graph data
        print(self._graph.nodes())

    def update_masks(self, train_mask: List[int], test_mask: List[int],
                     valid_mask: Optional[List[int]] = None) -> None:
        """
        Same function as CustomDataset parent to which we add a graph construction component

        Args:
            train_mask: list of idx to use for training
            test_mask: list of idx to use for test
            valid_mask: list of idx to use for validation

        Returns: None
        """
        CustomDataset.update_masks(self, train_mask, test_mask, valid_mask)
        self._update_graph()
