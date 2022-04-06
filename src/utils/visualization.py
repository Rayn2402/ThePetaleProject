"""
Filename: visualization.py

Authors: Nicolas Raymond

Description: This file contains all function related to data visualization

Date of last modification : 2021/03/29
"""

from matplotlib import pyplot as plt
from numpy import arange, array
from numpy import sum as npsum
from os.path import join
from sklearn.manifold import TSNE
from src.data.processing.datasets import MaskType
from src.recording.constants import MEAN, STD
from torch import tensor
from typing import Dict, List, Optional

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


def visualize_importance(data: Dict[str, Dict[str, float]],
                         figure_title: str,
                         filename: str) -> None:
    """
    Creates a bar plot with mean and standard deviations
    of variable importance contained within the dictionary

    Args:
        data: dictionary with variable name as keys and "mean" and "std" as values
        figure_title: name appearing over the plot
        filename: name of the file in which the figure is saved

    Returns: None
    """
    # We initialize three lists for the values, the errors, and the labels
    means, stds, labels = [], [], []

    # We collect the data of each hyperparameter importance
    for key in data.keys():
        mean = data[key][MEAN]
        if mean >= 0.01:
            means.append(mean)
            stds.append(data[key][STD])
            labels.append(key)

    # We sort the list according to their values
    sorted_means = sorted(means)
    sorted_labels = sorted(labels, key=lambda x: means[labels.index(x)])
    sorted_stds = sorted(stds, key=lambda x: means[stds.index(x)])

    # We build the plot
    y_pos = arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.barh(y_pos, sorted_means, xerr=sorted_stds, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Importance')
    ax.set_title(figure_title)

    # We save the plot
    plt.savefig(filename)
    plt.close()


def visualize_scaled_importance(data: Dict[str, Dict[str, float]],
                                figure_title: str,
                                filename: str) -> None:
    """
    Creates a bar plot with mean and standard deviations
    of variable importance contained within the dictionary

    Args:
        data: dictionary with variable name as keys and "mean" and "std" as values
        figure_title: name appearing over the plot
        filename: name of the file in which the figure is saved

    Returns: None
    """
    # We initialize two lists for the scaled values and the labels
    scaled_imp, labels = [], []

    # We collect the data of each hyperparameter importance
    for key in data.keys():
        mean = data[key][MEAN]
        if mean >= 0.01:
            scaled_imp.append(mean/(data[key][STD] + 0.001))
            labels.append(key)

    # We sort the list according values
    sorted_scaled_imp = sorted(scaled_imp)
    sorted_labels = sorted(labels, key=lambda x: scaled_imp[labels.index(x)])

    # We build the plot
    y_pos = arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.barh(y_pos, sorted_scaled_imp, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Importance')
    ax.set_title(figure_title)

    # We save the plot
    plt.savefig(filename)
    plt.close()
