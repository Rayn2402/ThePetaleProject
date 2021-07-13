"""
Authors : Nicolas Raymond

Files that contains class related to Datasets

"""

from dgl import heterograph
from numpy import array, concatenate
from pandas import DataFrame, Series
from src.data.extraction.constants import *
from src.data.processing.preprocessing import preprocess_continuous, preprocess_categoricals
from torch.utils.data import Dataset
from torch import from_numpy, tensor, cat
from typing import Optional, List, Callable, Tuple, Union, Any, Dict


class PetaleDataset(Dataset):
    """
    Scaffolding of all dataset classes implemented for our experiments
    """
    def __init__(self, df: DataFrame, target: str,
                 cont_cols: Optional[List[str]] = None, cat_cols: Optional[List[str]] = None,
                 classification: bool = True, to_tensor: bool = False):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            classification: True for classification task, False for regression
            to_tensor: True if we want the features and targets in tensors, False for numpy arrays

        """
        # Validations of inputs
        assert PARTICIPANT in df.columns, "Patients' ids missing from the dataframe."
        assert (cont_cols is not None or cat_cols is not None), "At least a list of continuous columns" \
                                                                " or a list of categorical columns must be given."
        for columns in [cont_cols, cat_cols]:
            self._check_columns_validity(df, columns)

        # Set default protected attributes
        self._classification = classification
        self._ids = list(df[PARTICIPANT].values)
        self._n = df.shape[0]
        self._original_data = df
        self._target = target
        self._to_tensor = to_tensor
        self._train_mask, self._valid_mask, self._test_mask = [], None, []
        self._x_cat, self._x_cont = None, None
        self._y = self._initialize_targets(df[target], classification, to_tensor)

        # Set default public attributes
        self.cont_cols, self.cont_idx = cont_cols, []
        self.cat_cols, self.cat_idx = cat_cols, []

        # Define protected feature "getter" method
        self._x = self._define_feature_getter(cont_cols, cat_cols, to_tensor)

        # We set a "getter" method to get modes of categorical columns and we also extract encodings
        self._get_modes, self._encodings = self._define_categorical_stats_getter(cat_cols)

        # We set a "getter" method to get mu ans std of continuous columns
        self._get_mu_and_std = self._define_numerical_stats_getter(cont_cols)

        # We set two "setter" methods to preprocess available data after masks update
        self._set_numerical = self._define_numerical_data_setter(cont_cols, to_tensor)
        self._set_categorical = self._define_categorical_data_setter(cat_cols, to_tensor)

        # We update current training mask with all the data
        self.update_masks(list(range(self._n)), [], [])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: Union[int, List[int]]) -> Tuple[Union[array, tensor], Union[array, tensor]]:
        return self.x[idx], self.y[idx]

    @property
    def classification(self) -> bool:
        return self._classification

    @property
    def cat_sizes(self) -> Optional[List[int]]:
        if self.encodings is not None:
            return [len(v.items()) for v in self.encodings.values()]
        return None

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
    def target(self) -> str:
        return self._target

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
        return self._x()

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
        x_cat, _ = preprocess_categoricals(self._original_data[self.cat_cols].copy(),
                                           mode=modes, encodings=self._encodings)
        self._x_cat = x_cat.to_numpy(dtype=int)

    def _define_categorical_data_setter(self, cat_cols: Optional[List[str]] = None,
                                        to_tensor: bool = False) -> Callable:
        """
        Defines the function used to set categorical data after masks update
        """
        if cat_cols is None:
            def set_categorical(modes: Optional[Series]) -> None:
                pass

            return set_categorical

        else:
            if to_tensor:
                def set_categorical(modes: Optional[Series]) -> None:
                    self._categorical_setter(modes)
                    self._x_cat = from_numpy(self._x_cat).long()

                return set_categorical

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

    def _define_feature_getter(self, cont_cols: Optional[List[str]] = None,
                               cat_cols: Optional[List[str]] = None, to_tensor: bool = False) -> Callable:
        """
        Defines the method used to extract features (processed data) available for training
        """

        if cont_cols is None:

            # Only categorical column idx
            self.cat_idx = list(range(len(cat_cols)))

            # Only categorical feature extracted by the getter
            def x() -> Union[tensor, array]:
                return self.x_cat

        elif cat_cols is None:

            # Only continuous column idx
            self.cont_idx = list(range(len(cont_cols)))

            # Only continuous features extracted by the getter
            def x() -> Union[tensor, array]:
                return self.x_cont

        else:
            # Continuous and categorical column idx
            nb_cont_cols = len(cont_cols)
            self.cont_idx = list(range(nb_cont_cols))
            self.cat_idx = [i + nb_cont_cols for i in range(len(cat_cols))]

            # Continuous and categorical features extracted by the getter
            if not to_tensor:
                def x() -> Union[tensor, array]:
                    return concatenate((self.x_cont, self.x_cat), axis=1)
            else:
                def x() -> Union[tensor, array]:
                    return cat((self.x_cont, self.x_cat), dim=1)

        return x

    def _define_numerical_data_setter(self, cont_cols: Optional[List[str]] = None,
                                      to_tensor: bool = False) -> Callable:
        """
        Defines the function used to set numerical continuous data after masks update
        """
        if cont_cols is None:
            def set_numerical(mu: Optional[Series], std: Optional[Series]) -> None:
                pass

            return set_numerical
        else:
            if to_tensor:
                def set_numerical(mu: Optional[Series], std: Optional[Series]) -> None:
                    self._numerical_setter(mu, std)
                    self._x_cont = from_numpy(self._x_cont).float()

                return set_numerical

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
        x_cont = preprocess_continuous(self._original_data[self.cont_cols].copy(), mu, std)

        # We apply the basis function
        self._x_cont = x_cont.to_numpy(dtype=float)

    def _retrieve_subset_from_original(self, cont_cols: Optional[List[str]] = None,
                                       cat_cols: List[str] = None) -> DataFrame:
        """
        Returns a copy of a subset of the original dataframe

        Args:
            cont_cols: list of continuous columns
            cat_cols: list of categorical columns

        Returns: dataframe
        """
        selected_cols = []
        if cont_cols is not None:
            selected_cols += cont_cols
        if cat_cols is not None:
            selected_cols += cat_cols

        return self.original_data[[PARTICIPANT, self._target] + selected_cols].copy()

    def get_imputed_dataframe(self):
        """
        Returns a copy of the original pandas dataframe where missing values are imputed according to
        the training mask.
        """
        imputed_df = self.original_data.drop([PARTICIPANT, self.target], axis=1).copy()
        if self.cont_cols is not None:
            imputed_df[self.cont_cols] = self._x_cont
        if self.cat_cols is not None:
            imputed_df[self.cat_cols] = self._x_cat

        return imputed_df

    def create_subset(self, cont_cols: Optional[List[str]] = None, cat_cols: List[str] = None) -> Any:
        """
        Returns a subset of the current dataset using the given cont_cols and cat_cols

        Args:
            cont_cols: list of continuous columns
            cat_cols: list of categorical columns

        Returns: instance of the same class
        """
        subset = self._retrieve_subset_from_original(cont_cols, cat_cols)
        return PetaleDataset(subset, self.target, cont_cols, cat_cols, self.classification, self._to_tensor)

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


class PetaleGNNDataset(PetaleDataset):
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
        super().__init__(df, target, cont_cols, cat_cols, classification, to_tensor=True)

    def _update_graph(self) -> None:
        """
        Updates the graph structures and its data once the masks are updated

        Returns: None
        """
        # We look through categorical columns to generate graph structure
        imp_df = self.get_imputed_dataframe()
        graph_structure = {}
        for e_types, e_values in self.encodings.items():
            edges_start, edges_end = [], []
            for value in e_values.values():
                idx_subset = imp_df.loc[imp_df[e_types] == value].index.to_numpy()
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
        PetaleDataset.update_masks(self, train_mask, test_mask, valid_mask)
        self._update_graph()
