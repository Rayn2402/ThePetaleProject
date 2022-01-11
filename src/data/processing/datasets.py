"""
Filename: datasets.py

Author: Nicolas Raymond

Description: Defines the classes related to datasets

Date of last modification : 2021/11/02
"""

from dgl import DGLGraph, DGLHeteroGraph, graph, heterograph, node_subgraph
from numpy import array, concatenate, where
from pandas import DataFrame, merge, Series
from src.data.extraction.constants import *
from src.data.processing.preprocessing import preprocess_categoricals, preprocess_continuous
from torch.utils.data import Dataset
from torch import cat, from_numpy, tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class MaskType:
    """
    Stores the constant related to mask types
    """
    TRAIN: str = "train"
    VALID: str = "valid"
    TEST: str = "test"
    INNER: str = "inner"

    def __iter__(self):
        return iter([self.TRAIN, self.VALID, self.TEST])


class PetaleDataset(Dataset):
    """
    Custom dataset class for Petale experiments
    """
    def __init__(self,
                 df: DataFrame,
                 target: str,
                 cont_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 gene_cols: Optional[List[str]] = None,
                 classification: bool = True,
                 to_tensor: bool = False):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            gene_cols: list of categorical column names that must be considered as genes
            classification: true for classification task, false for regression
            to_tensor: true if we want the features and targets in tensors, false for numpy arrays

        """
        # Validations of inputs
        if PARTICIPANT not in df.columns:
            raise ValueError("Patients' ids missing from the dataframe")

        if cont_cols is None and cat_cols is None:
            raise ValueError("At least a list of continuous columns or a list of categorical columns must be provided.")

        for columns in [cont_cols, cat_cols]:
            self._check_columns_validity(df, columns)

        if gene_cols is not None:
            self._gene_cols = gene_cols
            self._check_genes_validity(cat_cols, gene_cols)
        else:
            self._gene_cols = []

        # Set default protected attributes
        self._cat_cols, self._cat_idx = cat_cols, []
        self._classification = classification
        self._cont_cols, self._cont_idx = cont_cols, []
        self._ids = list(df[PARTICIPANT].values)
        self._ids_to_row_idx = {id_: i for i, id_ in enumerate(self._ids)}
        self._n = df.shape[0]
        self._original_data = df
        self._target = target
        self._to_tensor = to_tensor
        self._train_mask, self._valid_mask, self._test_mask = [], None, []
        self._x_cat, self._x_cont = None, None
        self._y = self._initialize_targets(df[target], classification, to_tensor)

        # Define protected feature "getter" method
        self._x = self._define_feature_getter(cont_cols, cat_cols, to_tensor)

        # Set attribute associated to genes idx
        self._gene_idx = {c: self._cat_idx[self.cat_cols.index(c)] for c in self._gene_cols}
        self._cat_idx_without_genes = [i for i in self._cat_idx if i not in self._gene_idx.values()]
        self._gene_idx_groups = self._create_genes_idx_group()

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

    def __getitem__(self, idx: Union[int, List[int]]
                    ) -> Union[Tuple[array, array, array], Tuple[tensor, tensor, tensor]]:
        return self.x[idx], self.y[idx], idx

    @property
    def classification(self) -> bool:
        return self._classification

    @property
    def cat_cols(self) -> List[str]:
        return self._cat_cols

    @property
    def cat_idx(self) -> List[int]:
        return self._cat_idx_without_genes

    @property
    def cat_sizes(self) -> Optional[List[int]]:
        if self._encodings is not None:
            return [len(self._encodings[c].items()) for c in self._cat_cols if c not in self._gene_cols]
        return None

    @property
    def cont_cols(self) -> List[str]:
        return self._cont_cols

    @property
    def cont_idx(self) -> List[int]:
        return self._cont_idx

    @property
    def encodings(self) -> Dict[str, Dict[str, int]]:
        return self._encodings

    @property
    def gene_idx_groups(self) -> Dict[str, List[int]]:
        return self._gene_idx_groups

    @property
    def ids(self) -> List[str]:
        return self._ids

    @property
    def ids_to_row_idx(self) -> Dict[str, int]:
        return self._ids_to_row_idx

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
        x_cat, _ = preprocess_categoricals(self._original_data[self._cat_cols].copy(),
                                           mode=modes, encodings=self._encodings)

        self._x_cat = x_cat.to_numpy(dtype=int)

    def _define_categorical_data_setter(self,
                                        cat_cols: Optional[List[str]] = None,
                                        to_tensor: bool = False) -> Callable:
        """
        Defines the function used to set categorical data after masks update

        Args:
            cat_cols: list with names of categorical columns
            to_tensor: true if we want the data to be converted into tensor

        Returns: function
        """
        # If there is no categorical columns
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

    def _define_categorical_stats_getter(self,
                                         cat_cols: Optional[List[str]] = None
                                         ) -> Tuple[Callable, Dict[str, Dict[str, int]]]:
        """
        Defines the function used to extract the modes of categorical columns

        Args:
            cat_cols: list of categorical column names

        Returns: function, dictionary with categories encodings
        """
        # If there is not categorical column
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

    def _define_feature_getter(self,
                               cont_cols: Optional[List[str]] = None,
                               cat_cols: Optional[List[str]] = None,
                               to_tensor: bool = False) -> Callable:
        """
        Defines the method used to extract the features (processed data) for training

        Args:
            cont_cols: list of continuous column names
            cat_cols: list of categorical column names
            to_tensor: true if the data must be converted to tensor

        Returns: function
        """
        if cont_cols is None:

            # Only categorical column idx
            self._cat_idx = list(range(len(cat_cols)))

            # Only categorical feature extracted by the getter
            def x() -> Union[tensor, array]:
                return self.x_cat

        elif cat_cols is None:

            # Only continuous column idx
            self._cont_idx = list(range(len(cont_cols)))

            # Only continuous features extracted by the getter
            def x() -> Union[tensor, array]:
                return self.x_cont

        else:

            # Continuous and categorical column idx
            nb_cont_cols = len(cont_cols)
            self._cont_idx = list(range(nb_cont_cols))
            self._cat_idx = list(range(nb_cont_cols, nb_cont_cols + len(cat_cols)))

            # Continuous and categorical features extracted by the getter
            if not to_tensor:
                def x() -> Union[tensor, array]:
                    return concatenate((self.x_cont, self.x_cat), axis=1)
            else:
                def x() -> Union[tensor, array]:
                    return cat((self.x_cont, self.x_cat), dim=1)

        return x

    def _define_numerical_data_setter(self,
                                      cont_cols: Optional[List[str]] = None,
                                      to_tensor: bool = False) -> Callable:
        """
        Defines the function used to set numerical continuous data after masks update

        Args:
            cont_cols: list of continuous column names
            to_tensor: true if data needs to be converted into tensor

        Returns: function
        """
        # If there is no continuous column
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
        Defines the function used to extract the mean and the standard deviations
        of numerical columns in a dataframe.

        Args:
            cont_cols: list with names of continuous columns

        Returns: function
        """
        # If there is no continuous column
        if cont_cols is None:

            def get_mu_and_std(df: DataFrame) -> Tuple[None, None]:
                return None, None
        else:

            # Make sure that numerical data in the original dataframe is in the correct format
            self._original_data[cont_cols] = self._original_data[cont_cols].astype(float)

            def get_mu_and_std(df: DataFrame) -> Tuple[Series, Series]:
                return df[self._cont_cols].mean(), df[self._cont_cols].std()

        return get_mu_and_std

    def _get_augmented_dataframe(self,
                                 data: DataFrame,
                                 categorical: bool = False
                                 ) -> Tuple[DataFrame, Optional[List[str]], Optional[List[str]]]:
        """
        Returns an augmented dataframe by concatenating original df and data

        Args:
            data: pandas dataframe with 2 columns
                  First column must be PARTICIPANT ids
                  Second column must be the feature we want to add
            categorical: True if the new feature is categorical

        Returns: pandas dataframe, list of cont cols, list of cat cols
        """
        # Extraction of the original dataframe
        df = self._retrieve_subset_from_original(self._cont_cols, self._cat_cols)

        # We add the new feature
        df = merge(df, data, on=[PARTICIPANT], how=INNER)

        # We update the columns list
        feature_name = [f for f in data.columns if f != PARTICIPANT]
        if categorical:
            cat_cols = self._cat_cols + feature_name if self._cat_cols is not None else [feature_name]
            cont_cols = self._cont_cols
        else:
            cont_cols = self._cont_cols + feature_name if self._cont_cols is not None else [feature_name]
            cat_cols = self._cat_cols

        return df, cont_cols, cat_cols

    def _numerical_setter(self,
                          mu: Series,
                          std: Series) -> None:
        """
        Fills missing values of numerical continuous data according according to the means of the
        training mask and then normalizes continuous data using the means and the standard
        deviations of the training mask.

        Args:
            mu: means of the numerical column according to the training mask
            std: standard deviations of the numerical column according to the training mask

        Returns: None
        """
        # We fill missing with means and normalize the data
        x_cont = preprocess_continuous(self._original_data[self._cont_cols].copy(), mu, std)

        # We apply the basis function
        self._x_cont = x_cont.to_numpy(dtype=float)

    def _retrieve_subset_from_original(self,
                                       cont_cols: Optional[List[str]] = None,
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

    def _create_genes_idx_group(self) -> Optional[Dict[str, List[int]]]:
        """
        Regroup genes idx column by chromosome

        Returns:  dictionary where keys are names of chromosomes and values
                  are list of idx referring to columns of genes associated to
                  the chromosome
        """
        if len(self._gene_cols) == 0:
            return None

        gene_idx_groups = {}
        for chrom_pos in self._gene_cols:
            chrom = chrom_pos.split('_')[0]
            if chrom in gene_idx_groups:
                gene_idx_groups[chrom].append(self._gene_idx[chrom_pos])
            else:
                gene_idx_groups[chrom] = [self._gene_idx[chrom_pos]]

        return gene_idx_groups

    def get_imputed_dataframe(self) -> DataFrame:
        """
        Returns a copy of the original pandas dataframe where missing values
        are imputed according to the training mask.

        Returns: pandas dataframe
        """
        imputed_df = self.original_data.drop([PARTICIPANT, self.target], axis=1).copy()
        if self._cont_cols is not None:
            imputed_df[self._cont_cols] = array(self._x_cont)
        if self._cat_cols is not None:
            imputed_df[self._cat_cols] = array(self._x_cat)

        return imputed_df

    def build_homogeneous_population_graph(self, cat_cols: Optional[List[str]] = None) -> DGLGraph:
        """
        Builds an undirected homogeneous graph from the categorical columns mentioned

        Args:
            cat_cols: list of categorical columns

        Returns: Undirected homogeneous graph
        """
        # We make sure that given categorical columns are ok
        if cat_cols is not None:
            for c in cat_cols:
                if c not in self._cat_cols:
                    raise ValueError(f"Unrecognized categorical column name : {c}")
        else:
            cat_cols = self._cat_cols

        # We extract imputed dataframe but reinsert nan values into categorical column that were imputed
        df = self.get_imputed_dataframe()
        na_row_idx, na_col_idx = where(self.original_data[self._cat_cols].isna().to_numpy())
        for i, j in zip(na_row_idx, na_col_idx):
            df.iloc[i, j] = nan

        # We look through categorical columns to generate graph structure
        u, v = [], []
        for e_type, e_values in self._encodings.items():
            if e_type in cat_cols:
                for value in e_values.values():
                    u, v = self._get_graphs_edges(u=u, v=v, df=df, e_type=e_type, value=value)

        # We turn u,v into tensors
        u, v = tensor(u).long(), tensor(v).long()

        return graph((u, v))

    def create_subset(self,
                      cont_cols: Optional[List[str]] = None,
                      cat_cols: List[str] = None) -> Any:
        """
        Returns a subset of the current dataset using the given cont_cols and cat_cols

        Args:
            cont_cols: list of continuous columns
            cat_cols: list of categorical columns

        Returns: instance of the PetaleDataset class
        """
        subset = self._retrieve_subset_from_original(cont_cols, cat_cols)
        gene_cols = None if len(self._gene_cols) == 0 else [c for c in self._gene_cols if c in cat_cols]
        return PetaleDataset(df=subset,
                             target=self.target,
                             cont_cols=cont_cols,
                             cat_cols=cat_cols,
                             gene_cols=gene_cols,
                             classification=self.classification,
                             to_tensor=self._to_tensor)

    def create_superset(self,
                        data: DataFrame,
                        categorical: bool = False) -> Any:
        """
        Returns a superset of the current dataset by including the given data

        Args:
            data: pandas dataframe with 2 columns
                  First column must be PARTICIPANT ids
                  Second column must be the feature we want to add
            categorical: True if the new feature is categorical

        Returns: instance of the PetaleDataset class
        """
        # We build the augmented dataframe
        df, cont_cols, cat_cols = self._get_augmented_dataframe(data, categorical)

        return PetaleDataset(df=df,
                             target=self.target,
                             cont_cols=cont_cols,
                             cat_cols=cat_cols,
                             classification=self.classification,
                             to_tensor=self._to_tensor)

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

    def update_masks(self,
                     train_mask: List[int],
                     test_mask: List[int],
                     valid_mask: Optional[List[int]] = None) -> None:
        """
        Updates the train, valid and test masks and then preprocesses the data available
        according to the current statistics of the training data

        Args:
            train_mask: list of idx in the training set
            test_mask: list of idx in the test set
            valid_mask: list of idx in the valid set

        Returns: None
        """

        # We set the new masks values
        self._train_mask, self._test_mask = train_mask, test_mask
        self._valid_mask = valid_mask if valid_mask is not None else []

        # We compute the current values of mu, std, modes and encodings
        mu, std, modes = self.current_train_stats()

        # We update the data that will be available via __get_item__
        self._set_numerical(mu, std)
        self._set_categorical(modes)

    @staticmethod
    def _initialize_targets(targets_column: Series,
                            classification: bool,
                            target_to_tensor: bool) -> Union[array, tensor]:
        """
        Sets the targets according to the task and the choice of container

        Args:
            targets_column: column of the dataframe with the targets
            classification: true for classification task, false for regression
            target_to_tensor: true if we want the targets to be in a tensor, false for numpy array

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
    def _check_columns_validity(df: DataFrame,
                                columns: Optional[List[str]] = None) -> None:
        """
        Checks if the columns are all in the dataframe

        Args:
            df: pandas dataframe with original data
            columns: list of column names
        """
        if columns is not None:
            dataframe_columns = list(df.columns.values)
            for c in columns:
                if c not in dataframe_columns:
                    raise ValueError(f"Column {c} is not part of the given dataframe")

    @staticmethod
    def _check_genes_validity(cat_cols: Optional[List[str]],
                              gene_cols: List[str]) -> None:
        """
        Checks if all column names related to genes are included in categorical columns

        Args:
            cat_cols: list of categorical column names
            gene_cols: list of categorical columns related to genes

        Returns: None
        """
        cat_cols = [] if cat_cols is None else cat_cols
        for gene in gene_cols:
            if gene not in cat_cols:
                raise ValueError(f'Gene {gene} from gene_cols cannot be found in cat_cols')

    @staticmethod
    def _get_graphs_edges(u: List[int],
                          v: List[int],
                          df: DataFrame,
                          e_type: str,
                          value: int) -> Tuple[List[int], List[int]]:
        """
        Finds pairs of index in a dataframe that shares the same category "value" for
        the categorical column "e_type".

        Args:
            u: list of idx (representing nodes numbers)
            v: list of idx (representing nodes numbers)
            df: pandas dataframe
            e_type: column name (edge type)
            value: category value shared

        Returns: updated list of nodes u, updated list of nodes v
        """

        # We retrieve idx of patients having the same value for a categorical column
        idx_subset = df.loc[df[e_type] == value].index.to_numpy()
        subset_size = idx_subset.shape[0]

        # For patient with common value we add edges in both direction
        for i in range(subset_size):
            u += [idx_subset[i]] * (subset_size - 1)
            remaining_idx = list(range(i)) + list(range(i + 1, subset_size))
            v += list(idx_subset[remaining_idx])

        return u, v


class PetaleStaticGNNDataset(PetaleDataset):
    """
    Dataset used to train, valid and test Graph Neural Network.
    Static means that the edges are based only on the non null categorical values from the
    original dataframe. Hence, the structure of the graph does not change after masks update (ie. imputation).
    """

    def __init__(self,
                 df: DataFrame,
                 target: str,
                 cont_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 classification: bool = True):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            classification: true for classification task, False for regression

        """
        # Sets train, valid and test subgraphs data to default value
        self._subgraphs = {MaskType.TRAIN: tuple(), MaskType.VALID: tuple(), MaskType.TEST: tuple()}

        # We use the _init_ of the parent class
        super().__init__(df=df,
                         target=target,
                         cont_cols=cont_cols,
                         cat_cols=cat_cols,
                         classification=classification,
                         to_tensor=True)

        # We initialize the graph attribute proper to StaticGNNDataset class
        self._graph = self._build_heterogeneous_population_graph()

    @property
    def graph(self) -> DGLHeteroGraph:
        return self._graph

    @property
    def train_subgraph(self) -> Tuple[DGLHeteroGraph, List[int], Dict[int, int]]:
        return self._subgraphs[MaskType.TRAIN]

    @property
    def test_subgraph(self) -> Tuple[DGLHeteroGraph, List[int], Dict[int, int]]:
        return self._subgraphs[MaskType.TEST]

    @property
    def valid_subgraph(self) -> Tuple[DGLHeteroGraph, List[int], Dict[int, int]]:
        return self._subgraphs[MaskType.VALID]

    def _build_heterogeneous_population_graph(self) -> DGLHeteroGraph:
        """
        Builds the graph structure

        Returns: Heterogeneous graph representing the datasets with categorical columns as links
        """
        # We extract imputed dataframe but reinsert nan values into categorical column that were imputed
        df = self.get_imputed_dataframe()
        na_row_idx, na_col_idx = where(self.original_data[self._cat_cols].isna().to_numpy())
        for i, j in zip(na_row_idx, na_col_idx):
            df.iloc[i, j] = nan

        # We look through categorical columns to generate graph structure
        graph_structure = {}
        for e_type, e_values in self._encodings.items():
            u, v = [], []
            for value in e_values.values():
                self._get_graphs_edges(u=u, v=v, df=df, e_type=e_type, value=value)
                graph_structure[(PARTICIPANT, e_type, PARTICIPANT)] = (tensor(u).long(), tensor(v).long())

        return heterograph(graph_structure)

    def _set_subgraphs_data(self) -> None:
        """
        Sets subgraphs data after masks update

        Returns: None
        """
        # Set the subgraph associated to training set and a map matching each training
        # index to its physical position in the train mask
        self._subgraphs[MaskType.TRAIN] = (*self.get_arbitrary_subgraph(idx=self.train_mask), self.train_mask)

        # Set the subgraph associated to test and a map matching each test
        # index to its physical position in the train + test mask
        train_test_mask = self.train_mask + self.test_mask
        self._subgraphs[MaskType.TEST] = (*self.get_arbitrary_subgraph(idx=train_test_mask), train_test_mask)

        if len(self.valid_mask) != 0:

            # Set the subgraph associated to validation and a map matching each valid
            # index to its physical position in the train + valid mask
            train_valid_mask = self.train_mask + self.valid_mask
            self._subgraphs[MaskType.VALID] = (*self.get_arbitrary_subgraph(idx=train_valid_mask), train_valid_mask)

    def get_arbitrary_subgraph(self, idx: List[int]) -> Tuple[DGLHeteroGraph, Dict[int, int]]:
        """
        Returns a tuple with :
        1 - heterogeneous subgraph with only nodes associated to idx in the list
        2-  dictionary mapping each idx to each position in the list

        Args:
            idx: list of idx such as masks

        Returns: heterogeneous graph
        """
        return node_subgraph(self.graph, nodes=idx, store_ids=True), {v: i for i, v in enumerate(idx)}

    def get_metapaths(self) -> List[List[str]]:
        """
        Return list of metapaths that can relate patients together.
        In our case, metapaths are juste edges types (categorical columns)

        Returns: list of list with edges types
        """
        return [[key] for key in self._encodings.keys()]

    def update_masks(self,
                     train_mask: List[int],
                     test_mask: List[int],
                     valid_mask: Optional[List[int]] = None) -> None:
        """
        First, updates the train, valid and test masks and preprocess the data available
        according to the current statistics of the training data.

        Second, updates train, valid and test subgraph and idx map.

        """
        # We first update masks as usual for datasets
        PetaleDataset.update_masks(self, train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)

        # If we are not calling update_masks for initialization purpose
        if len(test_mask) != 0:
            self._set_subgraphs_data()

    def create_subset(self,
                      cont_cols: Optional[List[str]] = None,
                      cat_cols: List[str] = None) -> Any:
        """
        Returns a subset of the current dataset using the given cont_cols and cat_cols

        Args:
            cont_cols: list of continuous columns
            cat_cols: list of categorical columns

        Returns: instance of the PetaleStaticGNNDataset class
        """
        subset = self._retrieve_subset_from_original(cont_cols, cat_cols)
        return PetaleStaticGNNDataset(df=subset,
                                      target=self.target,
                                      cont_cols=cont_cols,
                                      cat_cols=cat_cols,
                                      classification=self.classification)

    def create_superset(self,
                        data: DataFrame,
                        categorical: bool = False) -> Any:
        """
        Returns a superset of the current dataset by including the given data

        Args:
            data: pandas dataframe with 2 columns
                  First column must be PARTICIPANT ids
                  Second column must be the feature we want to add
            categorical: True if the new feature is categorical

        Returns: instance of the PetaleStaticGNNDataset class
        """
        # We build the augmented dataframe
        df, cont_cols, cat_cols = self._get_augmented_dataframe(data, categorical)

        return PetaleStaticGNNDataset(df=df,
                                      target=self.target,
                                      cont_cols=cont_cols,
                                      cat_cols=cat_cols,
                                      classification=self.classification)
