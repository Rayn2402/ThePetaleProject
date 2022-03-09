"""
Filename: graph.py

Authors: Nicolas Raymond

Description: This file is used to define a class called PetaleGraph
             that was used to test CorrectAndSmooth algorithm.

Date of last modification : 2022/02/21
"""

from numpy import cov
from numpy.linalg import inv
from src.data.processing.datasets import PetaleDataset, MaskType
from torch import tensor, zeros, transpose, diag, pow, mm, eye, is_tensor, ones, topk
from typing import Dict, List, Optional


class PetaleGraph:
    """
    Represents an homogeneous undirected graph using edges, degree and weights matrices
    """
    def __init__(self,
                 dataset: PetaleDataset,
                 cat_cols: Optional[List[str]] = None,
                 include_distances: bool = False,
                 max_degree: Optional[int] = None):
        """
        Sets the edges, adjacency, weight and degrees matrices

        Args:
            dataset: custom dataset with training and test data
            cat_cols: list of categorical columns from which to build the graph, if None
                      all nodes are connected together.
            include_distances: true if we want to use distance between patient to define edges weights
            max_degree: maximum number of degree that a node can have, if None there is no limit.
        """
        
        self.edges = self.build_edges_matrix(dataset, cat_cols)
        self.degrees = self.edges.sum(dim=1).squeeze()

        if include_distances:

            # We first set the weights as 1/distance
            self.weight = 1/(self.compute_distances(dataset) + eye(self.edges.shape[0])) - eye(self.edges.shape[0])

            # We multiply the weights by the edges matrix to remove some connections
            self.weight *= self.edges

            # We normalize the weights on each row
            self.weight /= self.weight.sum(dim=1).reshape(-1, 1)

        else:
            self.weight = mm(diag(pow(self.degrees, -1)), self.edges)  # Inverse of degrees multiplied by edges

        # If a max degree is given we only keep the "max_degree" closest neighbors
        if max_degree is not None:

            # We find the top_k idx for each rows
            _, top_k_idx = topk(self.weight, k=max_degree, dim=1)

            # We create a filter for top_k neighbors of each row
            top_k_mask = zeros((len(dataset), len(dataset)))
            for i in range(top_k_idx.shape[0]):
                top_k_mask[i, top_k_idx[i, :]] = 1

            # We multiply weights by the filter (Hadamard product)
            self.weight *= top_k_mask

            # We normalize the weights on each row
            self.weight /= self.weight.sum(dim=1).reshape(-1, 1)

    def propagate_labels(self,
                         labels: tensor,
                         r: float,
                         nb_iter: int):
        """
        Applies iterative label propagation algorithm

        Args:
            labels: (N, ) tensor with ground truth
            r: proportion attributed to original labels throughout iteration
            nb_iter: nb of propagation iteration

        Returns: (N, ) tensor with new labels after propagation
        """
        if not (0 <= r <= 1):
            raise ValueError("r must be in range [0, 1]")

        final_labels = labels
        for i in range(nb_iter):
            final_labels = r*labels + (1-r)*mm(self.weight, final_labels)

        return final_labels

    def random_walk_with_restart(self,
                                 walk_length: int,
                                 r: float) -> tensor:
        """
        Approximates stationary distribution of each node using a random walk
        with restart of a given length.

        Args:
            walk_length: number of iteration during the random walk
            r: probability of restarting at the same node

        Returns: (N,N) tensor with approximate stationary distribution
        """
        # We initialize the matrices with stationary distributions as identity
        p_init = eye(self.edges.shape[0])  # initial distribution
        p = eye(self.edges.shape[0])       # stationary distribution

        # We proceed to random walk iteration
        for i in range(walk_length):
            p = r*p_init + (1 - r)*mm(p, self.weight)

        return p

    @staticmethod
    def build_edges_matrix(dataset: PetaleDataset,
                           cat_cols: Optional[List[str]] = None) -> tensor:
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
                if c not in dataset.cat_cols:
                    raise ValueError(f"Unrecognized categorical column name : {c}")

        # Otherwise, each node is connect to all the others
        else:
            return ones(size=(len(dataset), len(dataset))) - eye(len(dataset))

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

    @staticmethod
    def compute_distances(dataset: PetaleDataset) -> tensor:
        """
        Calculates mahalanobis distances between individuals according to
        the training set.

        Args:
            dataset: custom dataset with training and test data

        Returns: (N, N) tensor with distances
        """
        # We first compute inverse of covariance matrix related to numerical columns of training set
        numerical_data = dataset.x[:, dataset.cont_idx]
        numerical_train_data = numerical_data[dataset.train_mask, :]
        if is_tensor(numerical_train_data):
            numerical_train_data = numerical_train_data.numpy()
        inv_cov_mat = tensor(inv(cov(numerical_train_data, rowvar=False))).float()

        # We convert numerical data to tensor
        numerical_data = tensor(numerical_data).float()

        # We compute squared mahalanobis distances
        mahalanobis_dist = zeros(len(dataset), len(dataset), requires_grad=False)
        for i in range(len(dataset)):
            for j in range(i+1, len(dataset)):
                diff = (numerical_data[i, :] - numerical_data[j, :]).reshape(-1, 1)
                mahalanobis_dist[i, j] = mm(mm(transpose(diff, 0, 1), inv_cov_mat), diff)

        # We add the matrix to its transpose to have all distances
        mahalanobis_dist = mahalanobis_dist + transpose(mahalanobis_dist, 0, 1)

        return mahalanobis_dist


def correct_and_smooth(g: PetaleGraph,
                       pred: tensor,
                       labels: tensor,
                       masks: Dict[str, Optional[List[int]]],
                       r_correct: float,
                       r_smooth: float,
                       nb_iter: int) -> tensor:
    """
    Applies correction and smoothing to the original predictions

    Args:
        g: PetaleGraph
        pred: (N,) tensor with predictions
        labels: (N,) tensor with real labels (ground truth)
        masks: Dictionary with train, valid and test mask (at least train and test)
        r_correct: weight attributed to default errors calculated
        r_smooth: weight attributed to default labels
        nb_iter: nb of propagation iteration

    Returns: (N,) tensor with corrected and smoothed labels

    """
    if not (MaskType.TRAIN in masks.keys()) or not (MaskType.TEST in masks.keys()):
        raise ValueError(f"{MaskType.TRAIN} and {MaskType.TEST} must be in dictionary's keys")

    # We extract train and valid masks
    labeled_mask = masks[MaskType.TRAIN]
    if masks.get(MaskType.VALID) is not None:
        labeled_mask += masks[MaskType.VALID]

    # We propagate the errors
    labels = labels.reshape(-1, 1)
    errors = zeros(size=(len(labels), 1))
    errors[labeled_mask] = pred[labeled_mask] - labels[labeled_mask]
    errors = g.propagate_labels(errors, r_correct, nb_iter)

    # We smooth labels
    test_mask = masks[MaskType.TEST]
    labels[test_mask] = pred[test_mask] - errors[test_mask]
    labels = g.propagate_labels(labels, r_smooth, nb_iter)

    return labels.flatten()
