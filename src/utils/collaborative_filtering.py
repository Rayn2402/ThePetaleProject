"""
Filename: collaborative_filtering.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to store functions related to collaborative filtering

Date of last modification : 2021/11/01
"""
from torch import tensor, zeros, mm, topk
from typing import Optional


def run_collaborative_filtering(weights: tensor,
                                labels: tensor,
                                test_mask: tensor,
                                top_k: Optional[int] = None) -> tensor:
    """
    Predict the labels of idx in the test mask using a weighted average of others' labels

    Args:
        weights: (N,N) tensor with weights between patients
                 The i,j element is the weight of patient j according to i
        labels: (N,) tensor with ground truth
        test_mask: (N_prime,) tensor with idx associated to test set
        top_k: If k is not None, the K closest neighbors will be used for filtering
               otherwise, the weighted average will be calculated over all the training set

    Returns: (N,) tensor with predictions over all patients
    """
    # We set columns of the test set to zero
    n = weights.shape[0]
    for j in test_mask:
        weights[:, j] = zeros(n)

    # We find the k biggest neighbors on each row if a top_k is provided
    if top_k is not None:

        # We find the top_k idx for each rows
        top_k = min(top_k, len(test_mask))
        _, top_k_idx = topk(weights, k=top_k, dim=1)

        # We create a filter for top_k neighbors of each row
        top_k_mask = zeros((n, n))
        for i in range(top_k_idx.shape[0]):
            top_k_mask[i, top_k_idx[i, :]] = 1

        # We multiply weights by the filter (Hadamard product)
        weights = top_k_mask * weights

    # We normalize rows
    weights /= weights.sum(dim=1).reshape(-1, 1)

    # We compute predicted labels
    y_hat = mm(weights, labels.reshape(-1, 1)).squeeze()

    # We change predicted labels of train data to ground_truth
    train_mask = tensor([i for i in range(n) if i not in test_mask])
    y_hat[train_mask] = labels[train_mask]

    return y_hat

