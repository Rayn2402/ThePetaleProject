"""
Author : Nicolas Raymond

This file contains all function related to data visualization

"""
from typing import Union
from torch import tensor
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def compare_predictions(preds: tensor, targets: tensor, title: Union[str, None] = None) -> None:
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

    plt.xlabel('Ground truth')
    plt.ylabel('Predictions')
    plt.show()
    plt.close()


def visualize_embeddings(embeddings: tensor, category_levels: tensor,
                         perplexity: int = 10, title: Union[str, None] = None) -> None:
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
    else:
        plt.title('Embeddings visualization with TSNE')

    plt.show()
    plt.close()


def visualize_epoch_losses(train_loss_history: tensor, test_loss_history: tensor) -> None:
    """
    Visualizes train and test loss history over training epoch

    :param train_loss_history: (E,) tensor where E is the number of epochs
    :param test_loss_history: (E,) tensor
    """
    nb_epochs = train_loss_history.shape[0]
    if nb_epochs != test_loss_history.shape[0]:
        raise Exception("Both train and test tensors must be of the same shape")

    epochs = range(nb_epochs)

    plt.plot(epochs, train_loss_history, label='train loss')
    plt.plot(epochs, test_loss_history, label='test loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and test losses over epochs')
    plt.show()
    plt.close()


