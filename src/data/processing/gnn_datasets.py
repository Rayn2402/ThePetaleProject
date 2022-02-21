"""
Filename: gnn_datasets.py

Author: Nicolas Raymond

Description: File used to store datasets associated to GNNs

Date of last modification: 2022/02/21
"""

from dgl import DGLGraph, node_subgraph
from numpy import cov
from numpy.linalg import inv
from pandas import DataFrame
from src.data.extraction.constants import PARTICIPANT
from src.data.processing.datasets import MaskType, PetaleDataset
from torch import mm, tensor, transpose, zeros
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
                 classification: bool = True):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            k: maximum degree in the population graph built
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            gene_cols: list of categorical column names that must be considered as genes
            classification: true for classification task, False for regression

        """
        # We set train, valid and test subgraphs data to default value
        self._subgraphs = {MaskType.TRAIN: tuple(), MaskType.VALID: tuple(), MaskType.TEST: tuple()}

        # We save the number of k-nearest neigbors
        self._k = k

        # We use the _init_ of the parent class
        super().__init__(df=df,
                         target=target,
                         cont_cols=cont_cols,
                         cat_cols=cat_cols,
                         gene_cols=gene_cols,
                         classification=classification,
                         to_tensor=True)

        # We initialize the graph attribute proper to StaticGNNDataset class
        self._graph = self._build_population_graph()

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

    def _build_population_graph(self) -> DGLGraph:
        """
        Builds the graph structure

        Returns: Homogeneous graph representing the datasets
        """
        # We calculate 1/(1 + distance) - 1 between each item

        # We calculate the order the values in decreasing order for each item

        # For each item saves the k-nearest neighbors that are in the train set
        # or is own set

        # Build the graph

        raise NotImplemented

    def _compute_distances(self) -> tensor:
        """
        Calculates mahalanobis distances between individuals, using the training set
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
        mahalanobis_dist = mahalanobis_dist + transpose(mahalanobis_dist, 0, 1)

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

    def get_arbitrary_subgraph(self, idx: List[int]) -> Tuple[DGLGraph, Dict[int, int]]:
        """
        Returns a tuple with :
        1 - homogeneous subgraph with only the nodes associated to idx in the list
        2-  dictionary mapping each idx to its position in the list

        Args:
            idx: list of idx such as masks

        Returns: homogeneous graph
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
        raise NotImplementedError

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
        raise NotImplementedError
