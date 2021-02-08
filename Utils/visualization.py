"""
Author : Nicolas Raymond

This file contains all function related to data visualization

"""
from matplotlib import pyplot as plt


def compare_predictions(preds, targets, title=None):
    """
    Compares predictions to targets in a 2D scatter plot

    :param preds: tuple of (N,) tensors
    :param targets: (N,) tensor
    """
    if isinstance(preds, tuple):
        for pred in preds:
            plt.scatter(targets, pred)
    else:
        plt.scatter(targets, preds)

    if title is not None:
        plt.title(title)

    plt.show()
    plt.close()
