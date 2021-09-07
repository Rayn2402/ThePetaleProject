"""
Author : Nicolas Raymond

This file contains all function related to data visualization

"""
from typing import Union, Optional, List
from torch import tensor
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


def visualize_class_distribution(targets: array, label_names: dict, title: Optional[str] = None) -> None:
    """
    Shows a pie chart with classes distribution

    :param targets: Sequence of class targets
    :param label_names: Dictionary with names associated to target values
    :param title: Title for the plot
    """
    # We first count the number of instances of each value in the targets vector
    label_counts = {v: npsum(targets == k) for k, v in label_names.items()}

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


def visualize_epoch_progression(train_history: List[tensor], valid_history: List[tensor], progression_type: List[str],
                                path: str) -> None:
    """
    Visualizes train and test loss history over training epoch

    :param train_history: list of (E,) tensors where E is the number of epochs
    :param valid_history: list of (E,) tensor
    :param progression_type: list of string specifying thee type of the progressions to visualize
    :param path: (string) determines where to save the plots
    """
    plt.figure(figsize=(12, 8))
    if len(train_history) == 1:

        x = range(len(train_history[0]))
        plt.plot(x, train_history[0], label=f'train')
        plt.plot(x, valid_history[0], label=f'valid')

        plt.legend()

        plt.xlabel('Epochs')
        plt.ylabel(progression_type[0])
    else:
        for i in range(len(train_history)):

            nb_epochs = len(train_history[i])

            plt.subplot(1, 2, i+1)
            plt.plot(range(nb_epochs), train_history[i], label=f'train')
            if len(valid_history[i]) != 0:
                plt.plot(range(nb_epochs), valid_history[i], label=f'valid')

            plt.legend()

            plt.xlabel('Epochs')
            plt.ylabel(progression_type[i])

    plt.tight_layout()
    plt.savefig(join(path, "epochs_progression.png"))
    plt.close()
