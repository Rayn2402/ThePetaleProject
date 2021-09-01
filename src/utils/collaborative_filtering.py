"""
Author: Nicolas Raymond

This file store functions related to collaborative filtering
"""

from torch import tensor, zeros, mm


def run_collaborative_filtering(weights: tensor, labels: tensor, test_mask: tensor) -> tensor:
    """
    Predict the labels of idx in the test mask using a weighted average of others' labels
    Args:
        weights: (N,N) tensor with weights between patients
                 The i,j element is the weight of patient j according to i
        labels: (N,) tensor with ground truth
        test_mask: (N_prime,) tensor with idx associated to test set

    Returns: (N,) tensor with predictions over all patients
    """
    # We set columns of the test set to zero
    n = weights.shape[0]
    for j in test_mask:
        weights[:, j] = zeros(n)

    # We normalize rows
    weights /= weights.sum(dim=1)

    # We compute predicted labels
    y_hat = mm(weights, labels.reshape(-1, 1)).squeeze()

    # We change predicted labels of train data to ground_truth
    train_mask = tensor([i for i in range(n) if i not in test_mask])
    y_hat[train_mask] = labels[train_mask]

    return y_hat
