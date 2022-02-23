"""
Filename: gnn_datasets.py

Author: Nicolas Raymond

Description: File used to store datasets associated to GNNs

Date of last modification: 2022/02/23
"""
import matplotlib.pyplot as plt

from dgl import DGLGraph, graph, node_subgraph
from networkx import draw, connected_components
from numpy import cov
from numpy.linalg import inv
from pandas import DataFrame
from src.data.processing.datasets import MaskType, PetaleDataset
from torch import eye, mm, ones, tensor, topk, transpose, zeros
from typing import Any, Dict, List, Optional, Tuple


class PetaleKGNNDataset(PetaleDataset):
    """
    Dataset used to train, valid and test Graph Neural Network.
    K-GNN means that the graph structure is built using the K-nearest neighbors
    on the specified columns.
    """

    def __init__(self,
                 df: DataFrame,
                 target: str,
                 k: int = 5,
                 cont_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 gene_cols: Optional[List[str]] = None,
                 conditional_cat_col: Optional[str] = None,
                 classification: bool = True):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            k: number of closest neighbors used to build the population graph
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            gene_cols: list of categorical column names that must be considered as genes
            conditional_cat_col: name of column for which items need to share the same
                                 value in order to be allowed to be connected in the population graph
            classification: true for classification task, False for regression

        """
        # We save the conditional categorical column given
        if conditional_cat_col is not None:
            if conditional_cat_col not in cat_cols:
                raise ValueError(f'{conditional_cat_col} not found among cat_cols')
        self._conditional_cat_col = conditional_cat_col

        # We set the graph attribute to default value
        self._graph = None

        # We save the number of k-nearest neighbors
        self._k = k

        # We set train, valid and test subgraphs data to default value
        self._subgraphs = {MaskType.TRAIN: tuple(), MaskType.VALID: tuple(), MaskType.TEST: tuple()}

        # We use the _init_ of the parent class
        super().__init__(df=df,
                         target=target,
                         cont_cols=cont_cols,
                         cat_cols=cat_cols,
                         gene_cols=gene_cols,
                         classification=classification,
                         to_tensor=True)

    @property
    def graph(self) -> DGLGraph:
        return self._graph

    @property
    def train_subgraph(self) -> Tuple[DGLGraph, List[int], Dict[int, int]]:
        return self._subgraphs[MaskType.TRAIN]

    @property
    def test_subgraph(self) -> Tuple[DGLGraph, List[int], Dict[int, int]]:
        return self._subgraphs[MaskType.TEST]

    @property
    def valid_subgraph(self) -> Tuple[DGLGraph, List[int], Dict[int, int]]:
        return self._subgraphs[MaskType.VALID]

    def _build_neighbors_filter_mat(self) -> tensor:
        """
        Creates an (N,N) tensor where element i,j is a 1 if item-i and item-j
        shares the same value for the conditional_cat_col. Element i,j are zeros otherwise.

        Returns: (N,N) tensor
        """
        # If no conditional column was given
        if self._conditional_cat_col is None:
            return ones(self._n, self._n)

        # We initialize a matrix filled with zeros
        neighbors_filter = zeros(self._n, self._n)

        # We fill the upper triangle of the edges matrix
        df = self.get_imputed_dataframe()
        for value in self.encodings[self._conditional_cat_col].values():

            # Idx of patients sharing same categorical value
            idx_subset = df.loc[df[self._conditional_cat_col] == value].index.to_numpy()

            # For patient with common value we add 1
            k = 0
            for i in idx_subset:
                for j in idx_subset[k+1:]:
                    neighbors_filter[i, j] += 1
                k += 1

        # We add it to its transpose to have the complete matrix
        neighbors_filter = neighbors_filter + neighbors_filter.t()

        return neighbors_filter

    def _build_population_graph(self) -> DGLGraph:
        """
        Builds the graph structure

        Returns: Homogeneous graph representing the datasets
        """
        # We calculate similarities between each item (1/(1 + distance) - 1)
        similarities = 1/(self._compute_distances() + eye(self._n)) - eye(self._n)

        # We turn some similarities to zeros if a conditional column was given
        filter_mat = self._build_neighbors_filter_mat()
        similarities *= filter_mat

        # We count the number of ones in each row of the filter mat
        ones_count = (filter_mat == 1).sum(dim=1)

        # We get the idx of the (n-1)-closest neighbors of each item
        _, top_n_idx = topk(similarities, k=(self._n-1), dim=1)
        top_n_idx = top_n_idx.tolist()

        # For each element in the training set, we filter its top_n_idx list
        # to only keep element from the training set that are not 0's
        for i in self.train_mask:
            top_n_idx[i] = top_n_idx[i][:ones_count[i]]
            top_n_idx[i] = [j for j in top_n_idx[i] if j in self.train_mask]

        # For each element in the valid set, we filter its top_n_idx list
        # to only keep element from the its set or the training set that are not 0's
        for i in self.valid_mask:
            top_n_idx[i] = top_n_idx[i][:ones_count[i]]
            top_n_idx[i] = [j for j in top_n_idx[i] if j in self.train_mask + self.valid_mask]

        # For each element in the test set, we filter its top_n_idx list
        # to only keep element from the its set or the training set that are not 0's
        for i in self.test_mask:
            top_n_idx[i] = top_n_idx[i][:ones_count[i]]
            top_n_idx[i] = [j for j in top_n_idx[i] if j in self.train_mask + self.test_mask]

        # We build the edges of the graph
        u, v = [], []
        for i in range(len(top_n_idx)):
            nb_neighbor = min(len(top_n_idx[i]), self._k)
            u += top_n_idx[i][:nb_neighbor]
            v += [i]*nb_neighbor
        u, v = tensor(u).long(), tensor(v).long()

        # We return the graph
        return graph((u, v))

    def _compute_distances(self) -> tensor:
        """
        Calculates squared mahalanobis distances between individuals, using the training set
        to estimate covariance matrix

        Returns: (N, N) tensor with distances
        """
        # We first compute inverse of covariance matrix related to numerical columns of training set
        numerical_data = self.x[:, self.cont_idx]
        numerical_train_data = numerical_data[self.train_mask, :]
        inv_cov_mat = tensor(inv(cov(numerical_train_data, rowvar=False))).float()

        # We compute squared mahalanobis distances
        mahalanobis_dist = zeros(self._n, self._n, requires_grad=False)
        for i in range(self._n):
            for j in range(i+1, self._n):
                diff = (numerical_data[i, :] - numerical_data[j, :]).reshape(-1, 1)
                mahalanobis_dist[i, j] = mm(mm(transpose(diff, 0, 1), inv_cov_mat), diff)

        # We add the matrix to its transpose to have all distances
        mahalanobis_dist = mahalanobis_dist + mahalanobis_dist.t()

        return mahalanobis_dist

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

    # def draw_train_graph(self) -> None:
    #     """
    #     Draws the training graph using NetworkX
    #
    #     Returns: None
    #     """
    #     g, _, idx = self.train_subgraph
    #     draw(g.to_networkx(), node_color=self.y[idx])
    #     plt.show()

    def get_arbitrary_subgraph(self, idx: List[int]) -> Tuple[DGLGraph, Dict[int, int]]:
        """
        Returns a tuple with :
        1 - homogeneous subgraph with only the nodes associated to idx in the list
        2-  dictionary mapping each idx to its position in the list

        Args:
            idx: list of idx such as masks

        Returns: homogeneous graph, dict matching each training index to its physical position in idx list
        """
        return node_subgraph(self.graph, nodes=idx, store_ids=True), {v: i for i, v in enumerate(idx)}

    def update_masks(self,
                     train_mask: List[int],
                     test_mask: List[int],
                     valid_mask: Optional[List[int]] = None) -> None:
        """
        First, updates the train, valid and test masks and preprocess the data available
        according to the current statistics of the training data.

        Then, rebuild the graph by running k-nearest neighbors algorithm.

        Second, updates train, valid and test subgraph and idx map.

        """
        # We first update masks as usual for datasets
        PetaleDataset.update_masks(self, train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)

        # If we are not calling update_masks for initialization purpose
        if len(test_mask) != 0:

            # We update the graph structure
            self._graph = self._build_population_graph()

            # We update the subgraphs data
            self._set_subgraphs_data()

    def create_subset(self,
                      cont_cols: Optional[List[str]] = None,
                      cat_cols: List[str] = None) -> Any:
        """
        Returns a subset of the current dataset using the given cont_cols and cat_cols

        Args:
            cont_cols: list of continuous columns
            cat_cols: list of categorical columns

        Returns: instance of the PetaleKGNNDataset class
        """
        subset = self._retrieve_subset_from_original(cont_cols, cat_cols)
        gene_cols = None if len(self._gene_cols) == 0 else [c for c in self._gene_cols if c in cat_cols]
        return PetaleKGNNDataset(df=subset,
                                 target=self.target,
                                 cont_cols=cont_cols,
                                 cat_cols=cat_cols,
                                 gene_cols=gene_cols,
                                 conditional_cat_col=self._conditional_cat_col,
                                 classification=self.classification)

    def create_superset(self,
                        data: DataFrame,
                        categorical: bool = False,
                        gene: bool = False) -> Any:
        """
        Returns a superset of the current dataset by including the given data

        Args:
            data: pandas dataframe with 2 columns
                  First column must be PARTICIPANT ids
                  Second column must be the feature we want to add
            categorical: True if the new feature is categorical
            gene: True if the new feature is considered as a gene

        Returns: instance of the PetaleKGNNDataset class
        """
        # We build the augmented dataframe
        df, cont_cols, cat_cols, gene_cols = self._get_augmented_dataframe(data, categorical, gene)

        return PetaleKGNNDataset(df=df,
                                 target=self.target,
                                 cont_cols=cont_cols,
                                 cat_cols=cat_cols,
                                 gene_cols=gene_cols,
                                 conditional_cat_col=self._conditional_cat_col,
                                 classification=self.classification)
