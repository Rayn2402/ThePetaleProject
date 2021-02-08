"""
Author : Nicolas Raymond

This file contains all function related to data visualization

"""
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def compare_predictions(preds, targets, title=None):
    """
    Compares predictions to targets in a 2D scatter plot

    :param preds: tuple of (N,) tensors
    :param title: string to add as title of the plot
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


def visualize_embeddings(embeddings, category_levels, perplexity=10, title=None):
    """
    Visualizes embeddings in a 2D space

    :param embeddings: (NxD) tensor
    :param category_levels: (N,) tensor (with category indices)
    :param perplexity: (int) perplexity parameter of TSNE
    :param title: string to add as title of the plot
    """
    X = embeddings.numpy()
    y = category_levels.numpy()

    if X.shape[1] > 2:
        X = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], c=y)

    if title is not None:
        plt.title(title)

    plt.show()
    plt.close()

