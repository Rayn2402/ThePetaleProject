"""
Author: Nicolas Raymond

This file is used to store the PetaleGraph class
"""

from numpy import array, cov
from numpy.linalg import inv
from src.data.processing.datasets import PetaleDataset
from torch import tensor, zeros, transpose, diag, pow, mm, eye, is_tensor
from typing import List, Optional


class PetaleGraph:
    """
    Represents an homogeneous undirected graph
    """
    def __init__(self, dataset: PetaleDataset, cat_cols: Optional[List[str]] = None,
                 include_distances: bool = False):
        """
        Sets the edges, adjacency, weight and degrees matrix
        Args:
            dataset: custom dataset with training and test data
            cat_cols: list of categorical columns from which to build the graph
            include_distances: True if we want to use distance between patient to define edges weights
        """
        
        self.edges = self.build_edges_matrix(dataset, cat_cols)
        self.degrees = self.edges.sum(dim=1).squeeze()
        if include_distances:

            # We first set the weights as 1/distance
            self.weight = 1/(self.compute_distances(dataset) + eye(self.edges.shape[0])) - eye(self.edges.shape[0])

            # We normalize the weights on each row
            self.weight /= self.weight.sum(axis=1)

        else:
            self.weight = mm(diag(pow(self.degrees, -1)), self.edges)  # Inverse of degrees multiplied by edges

    def random_walk_with_restart(self, walk_length: int, restart_proba: float) -> tensor:
        """
        Approximate stationary distribution of each node using a random walk
        with restart of a given length.

        Args:
            walk_length: number of iteration during the random walk
            restart_proba: probability of restarting at the same node

        Returns: (N,N) tensor with approximate stationary distribution
        """
        # We initialize the matrices with stationary distributions as identity
        p_init = eye(self.edges.shape[0])  # initial distribution
        p = eye(self.edges.shape[0])       # stationary distribution

        # We proceed to random walk iteration
        for i in range(walk_length):
            p = restart_proba*p_init + (1 - restart_proba)*mm(p, self.weight)

        return p

    @staticmethod
    def compute_distances(dataset: PetaleDataset) -> tensor:
        """
        Calculates mahalanobis distances between individuals
        Args:
            dataset: custom dataset with training and test data

        Returns: (N, N) tensor with distances
        """
        # We first compute inverse of covariance matrix related to numerical columns
        x, _, _ = dataset[:]
        numerical_data = x[:, dataset.cont_idx]
        if is_tensor(numerical_data):
            numerical_data = numerical_data.numpy()
        cov_mat = tensor(inv(cov(numerical_data, rowvar=False))).float()

        # We convert numerical data to tensor
        numerical_data = tensor(numerical_data).float()

        # We compute squared mahalanobis distances
        mahalanobis_dist = zeros(len(dataset), len(dataset), requires_grad=False)
        for i in range(len(dataset)):
            for j in range(i+1, len(dataset)):
                diff = (numerical_data[i, :] - numerical_data[j, :]).reshape(-1, 1)
                mahalanobis_dist[i, j] = mm(mm(transpose(diff, 0, 1), cov_mat), diff)

        # We add the matrix to its transpose to have all distances
        mahalanobis_dist = mahalanobis_dist + transpose(mahalanobis_dist, 0, 1)

        return mahalanobis_dist

    @staticmethod
    def build_edges_matrix(dataset: PetaleDataset, cat_cols: Optional[List[str]] = None) -> tensor:
        """
        Creates a matrix that stores the number of edges between nodes
        Args:
            dataset: custom dataset with training and test data
            cat_cols: list of categorical columns from which to build the graph

        Returns: (N, N) tensor with number of edges between nodes
        """
        # Initialization of matrix filled with 0s
        edges = zeros(len(dataset), len(dataset), requires_grad=False)

        # We make sure that the given categorical columns are in the dataframe
        if cat_cols is not None:
            for c in cat_cols:
                assert c in dataset.cat_cols, f"Unrecognized categorical column name : {c}"
        else:
            cat_cols = dataset.cat_cols

        # We fill the upper triangle of the edges matrix
        df = dataset.original_data[cat_cols]
        for category, values in dataset.encodings.items():
            for value in values.keys():

                # Idx of patients sharing same categorical value
                idx_subset = df.loc[df[category] == value].index.to_numpy()

                # For patient with common value we add edges
                k = 0
                for i in idx_subset:
                    for j in idx_subset[k+1:]:
                        edges[i, j] += 1
                    k += 1

        # We add it to its transpose to have the complete matrix
        edges = edges + transpose(edges, 0, 1)

        return edges
