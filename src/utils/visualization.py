"""
Filename: visualization.py

Authors: Nicolas Raymond

Description: This file contains all function related to data visualization

Date of last modification : 2021/11/01
"""

from matplotlib import pyplot as plt
from numpy import array
from numpy import sum as npsum
from os.path import join
from sklearn.manifold import TSNE
from src.data.processing.datasets import MaskType
from torch import tensor
from typing import List, Optional

# Epochs progression figure name
EPOCHS_PROGRESSION_FIG: str = "epochs_progression.png"


def format_to_percentage(pct: float, values: List[float]) -> str:
    """
    Change a float to a str representing a percentage
    Args:
        pct: count related to a class
        values: count of items in each class

    Returns: str
    """
    absolute = int(round(pct/100.*npsum(values)))
    return "{:.1f}%".format(pct, absolute)


def visualize_class_distribution(targets: array,
                                 label_names: dict,
                                 title: Optional[str] = None) -> None:
    """
    Shows a pie chart with classes distribution

    Args:
        targets: array of class targets
        label_names: dictionary with names associated to target values
        title: title for the plot

    Returns: None
    """

    # We first count the number of instances of each value in the targets vector
    label_counts = {v: npsum(targets == k) for k, v in label_names.items()}

    # We prepare a list of string to use as plot labels
    labels = [f"{k} ({v})" for k, v in label_counts.items()]

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(label_counts.values(),
                                      textprops=dict(color="w"),
                                      startangle=90,
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


def visualize_embeddings(embeddings: tensor,
                         category_levels: tensor,
                         perplexity: int = 10,
                         title: Optional[str] = None) -> None:
    """
    Visualizes embeddings in a 2D space

    Args:
        embeddings: (N,D) tensor
        category_levels: (N,) tensor (with category indices)
        perplexity: perplexity parameter of TSNE
        title: title of the plot

    Returns: None
    """
    # Convert tensor to numpy array
    X = embeddings.numpy()
    y = category_levels.numpy()

    # If the embeddings have more than 2 dimensions, project them with TSNE
    if X.shape[1] > 2:
        X = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)

    # Create the plot
    plt.scatter(X[:, 0], X[:, 1], c=y)

    if title is not None:
        plt.title(title)
    else:
        plt.title('Embeddings visualization with TSNE')

    plt.show()
    plt.close()


def visualize_epoch_progression(train_history: List[tensor],
                                valid_history: List[tensor],
                                progression_type: List[str],
                                path: str) -> None:
    """
    Visualizes train and test loss histories over training epoch

    Args:
        train_history: list of (E,) tensors where E is the number of epochs
        valid_history: list of (E,) tensor
        progression_type: list of string specifying the type of the progressions to visualize
        path: path where to save the plots

    Returns: None
    """
    plt.figure(figsize=(12, 8))

    # If there is only one plot to show (related to the loss)
    if len(train_history) == 1:

        x = range(len(train_history[0]))
        plt.plot(x, train_history[0], label=MaskType.TRAIN)
        plt.plot(x, valid_history[0], label=MaskType.VALID)

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(progression_type[0])

    # If there are two plots to show (one for the loss and one for the evaluation metric)
    else:
        for i in range(len(train_history)):

            nb_epochs = len(train_history[i])
            plt.subplot(1, 2, i+1)
            plt.plot(range(nb_epochs), train_history[i], label=MaskType.TRAIN)
            if len(valid_history[i]) != 0:
                plt.plot(range(nb_epochs), valid_history[i], label=MaskType.VALID)

            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(progression_type[i])

    plt.tight_layout()
    plt.savefig(join(path, EPOCHS_PROGRESSION_FIG))
    plt.close()
