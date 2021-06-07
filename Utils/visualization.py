"""
Author : Nicolas Raymond

This file contains all function related to data visualization

"""
from typing import Union, Optional
from torch import tensor, sum, long
from matplotlib import pyplot as plt
from numpy import array
from numpy import sum as npsum
from sklearn.manifold import TSNE
from os.path import join


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


def format_to_percentage(pct, values):
    absolute = int(round(pct/100.*npsum(values)))
    return "{:.1f}%".format(pct, absolute)


def visualize_class_distribution(targets: Union[tensor, array], label_names: dict, title: Optional[str] = None) -> None:
    """
    Shows a pie chart with classes distribution

    :param targets: Sequence of class targets
    :param label_names: Dictionary with names associated to target values
    :param title: Title for the plot
    """
    # We first count the number of instances of each value in the targets vector
    label_counts = {v: sum(tensor(targets) == k) for k, v in label_names.items()}

    # We prepare a list of string to use as plot labels
    labels = [f"{k} ({v})" for k, v in label_counts.items()]

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(label_counts.values(), textprops=dict(color="w"), startangle=90,
                                      autopct=lambda pct: format_to_percentage(pct, list(label_counts.values())))
    ax.legend(wedges, labels,
              title="Labels",
              loc="center right",
              bbox_to_anchor=(0.1, 0.5, 0, 0),
              prop={"size": 8})

    plt.setp(autotexts, size=8, weight="bold")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


def visualize_embeddings(embeddings: tensor, category_levels: tensor,
                         perplexity: int = 10, title: Optional[str] = None) -> None:
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


def visualize_epoch_progression(train_history: tensor, valid_history: tensor, progression_type: str, path: str) -> None:
    """
    Visualizes train and test loss history over training epoch

    :param train_history: (E,) tensor where E is the number of epochs
    :param valid_history: (E,) tensor
    :param progression_type: (string) type of the progression to visualize
    :param path: (string) determines where to save the plots
    """
    nb_epochs = train_history.shape[0]
    if nb_epochs != valid_history.shape[0]:
        raise Exception("Both train and valid tensors must be of the same shape")

    epochs = range(nb_epochs)

    plt.plot(epochs, train_history, label=f'train {progression_type}')
    plt.plot(epochs, valid_history, label=f'valid {progression_type}')

    plt.legend()

    title = f'Train_and_valid_{progression_type}_over_epochs'

    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(join(path, f"{title}.png"))
    plt.close()
