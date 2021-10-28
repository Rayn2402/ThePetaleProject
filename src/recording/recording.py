"""
Filename: recording.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to define the Recorder class

Date of last modification : 2021/10/28
"""

import json
import matplotlib.pyplot as plt
import os
import pickle

from collections import Counter
from numpy import arange, max, mean, median, min, std
from src.models.abstract_models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.recording.constants import *
from torch import tensor, save
from torch.nn import Module
from typing import Any, Dict, List, Union


class Recorder:
    """
    Recorder objects used save results of the experiments
    """
    def __init__(self,
                 evaluation_name: str,
                 index: int,
                 recordings_path: str):
        """
        Sets protected attributes

        Args:
            evaluation_name: name of the evaluation
            index: index of the outer split
            recordings_path: path leading to where we want to save the results
        """

        # We store the protected attributes
        self._data = {NAME: evaluation_name,
                      INDEX: index,
                      DATA_INFO: {},
                      HYPERPARAMETERS: {},
                      HYPERPARAMETER_IMPORTANCE: {},
                      TRAIN_METRICS: {},
                      TEST_METRICS: {},
                      COEFFICIENT: {},
                      TRAIN_RESULTS: {},
                      TEST_RESULTS: {}}

        self._path = os.path.join(recordings_path, evaluation_name, f"Split_{index}")

        # We create the folder where the information will be saved
        os.makedirs(self._path, exist_ok=True)

    def generate_file(self) -> None:
        """
        Save the protected dictionary into a json file

        Returns: None
        """
        # We save all the data collected in a json file
        filepath = os.path.join(self._path, RECORDS_FILE)
        with open(filepath, "w") as file:
            json.dump(self._data, file, indent=True)

    def record_coefficient(self,
                           name: str,
                           value: float) -> None:
        """
        Saves the value associated to a coefficient (used for linear regression)

        Args:
            name: name of the variable associated to the coefficient
            value: value of the coefficient

        Returns: None
        """
        self._data[COEFFICIENT][name] = value

    def record_data_info(self,
                         data_name: str,
                         data: Any) -> None:
        """
        Records the specific value "data" associated to the variable "data_name" in
        the protected dictionary

        Args:
            data_name: name of the variable for which we want to save a specific information
            data: value we want to store

        Returns: None

        """
        self._data[DATA_INFO][data_name] = data

    def record_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Saves the hyperparameters in the protected dictionary

        Args:
            hyperparameters: dictionary of hyperparameters and their value

        Returns: None
        """
        # We save all the hyperparameters
        for key in hyperparameters.keys():
            self._data[HYPERPARAMETERS][key] = round(hyperparameters[key], 6) if \
                isinstance(hyperparameters[key], float) else hyperparameters[key]

    def record_hyperparameters_importance(self, hyperparameter_importance: Dict[str, float]) -> None:
        """
        Saves the hyperparameters' importance in the protected dictionary

        Args:
            hyperparameter_importance: dictionary of hyperparameters and their importance

        Returns: None
        """
        # We save all the hyperparameters importance
        for key in hyperparameter_importance.keys():
            self._data[HYPERPARAMETER_IMPORTANCE][key] = round(hyperparameter_importance[key], 4)

    def record_model(self, model: Union[PetaleBinaryClassifier, PetaleRegressor]) -> None:
        """
        Saves a model using pickle or torch's save function

        Args:
            model: model to save

        Returns: None

        """
        # If the model is a torch module with save it using torch
        if isinstance(model, Module):
            save(model, os.path.join(self._path, "model.pt"))
        else:
            # We save the model with pickle
            filepath = os.path.join(self._path, "model.sav")
            pickle.dump(model, open(filepath, "wb"))

    def record_scores(self,
                      score: float,
                      metric: str,
                      test: bool = True) -> None:
        """
        Saves the score associated to a metric

        Args:
            score: float
            metric: name of the metric
            test: true if the scores are recorded for the test set

        Returns: None

        """
        # We save the score of the given metric
        section = TEST_METRICS if test else TRAIN_METRICS
        self._data[section][metric] = round(score, 6)

    def record_predictions(self,
                           ids: List[str],
                           predictions: tensor,
                           target: tensor,
                           test: bool = True) -> None:
        """
        Save the predictions of a given model for each patient ids

        Args:
            ids: patient/participant ids
            predictions: predicted class or regression value
            target: target value
            test: true if the predictions are recorded for the test set

        Returns: None
        """
        # We save the predictions
        section = TEST_RESULTS if test else TRAIN_RESULTS
        if len(predictions.shape) == 0:
            for j, id_ in enumerate(ids):
                self._data[section][str(id_)] = {
                    PREDICTION: str(predictions[j].item()),
                    TARGET: str(target[j].item())}
        else:
            for j, id_ in enumerate(ids):
                self._data[section][str(id_)] = {
                    PREDICTION: str(predictions[j].tolist()),
                    TARGET: str(target[j].item())}


def get_evaluation_recap(evaluation_name: str, recordings_path: str) -> None:
    """
    Creates a file with a summary of results from records.json file of each data split

    Args:
        evaluation_name: name of the evaluation
        recordings_path: directory where containing the folders with the results of each split

    Returns: None
    """

    # We check if the directory with results exists
    path = os.path.join(recordings_path, evaluation_name)
    if not os.path.exists(path):
        raise ValueError('Impossible to find the given directory')

    # We sort the folders in the directory according to the split number
    folders = next(os.walk(path))[1]
    folders.sort(key=lambda x: int(x.split("_")[1]))

    # Initialization of an empty dictionary to store the summary
    data = {
        TRAIN_METRICS: {},
        TEST_METRICS: {},
        HYPERPARAMETER_IMPORTANCE: {},
        HYPERPARAMETERS: {},
        COEFFICIENT: {}
    }

    # Initialization of a list of key list that we can found within section of records dictionary
    key_lists = [None]*4

    for folder in folders:

        # We open the json file containing the info of each split
        with open(os.path.join(path, folder, RECORDS_FILE), "r") as read_file:
            split_data = json.load(read_file)

        # For each section and their respective key list
        for section, key_list in zip(data.keys(), key_lists):

            if section in split_data.keys():

                # If the key list is not initialized yet..
                if key_list is None:

                    # Initialization of the key list
                    key_list = split_data[section].keys()

                    # Initialization of each individual key section in the dictionary
                    for key in key_list:
                        data[section][key] = {VALUES: [], INFO: ""}

                # We add values to each key associated to the current section
                for key in key_list:
                    data[section][key][VALUES].append(split_data[section][key])

    # We add the info about the mean, the standard deviation, the median , the min, and the max
    set_info(data)

    # We save the json containing the summary of the records
    with open(os.path.join(path, SUMMARY_FILE), "w") as file:
        json.dump(data, file, indent=True)


def set_info(data: Dict[str, Dict[str, Union[List[Union[str, float]], str]]]) -> None:
    """
    Adds the mean, the standard deviation, the median, the min and the max
    to the numerical parameters of each section of the dictionary with the summary.

    Otherwise, counts the number of appearances of the categorical parameters.

    Args:
        data: dictionary with the summary of results from the splits' records

    Returns: None

    """
    # For each section
    for section in data.keys():

        # For each key of this section
        for key in data[section].keys():

            # We extract the list of values
            values = data[section][key][VALUES]

            if not isinstance(values[0], str):
                mean_, std_ = round(mean(values), 4), round(std(values), 4)
                data[section][key][INFO] = f"{mean_} +- {std_} [{median(values)}; {min(values)}-{max(values)}]"
                data[section][key][MEAN] = mean_
                data[section][key][STD] = std_
            else:
                counts = Counter(data[section][key][VALUES])
                data[section][key][INFO] = str(dict(counts))


def plot_hps_importance_chart(evaluation_name: str, recordings_path: str) -> None:
    """
    Creates a bar plot containing information about the mean and standard deviation
    of each hyperparameter's importance.

    Args:
        evaluation_name: name of the evaluation
        recordings_path: directory where containing the folders with the results of each split

    Returns: None

    """
    # We get the content of the json file
    path = os.path.join(recordings_path, evaluation_name)
    with open(os.path.join(path, SUMMARY_FILE), "r") as read_file:
        data = json.load(read_file)[HYPERPARAMETER_IMPORTANCE]

    # We initialize three lists for the values, the errors, and the labels
    values, errors, labels = [], [], []

    # We collect the data of each hyperparameter importance
    for key in data.keys():
        values.append(data[key][MEAN])
        errors.append(data[key][STD])
        labels.append(key)

    # We sort the list according values
    sorted_values = sorted(values)
    sorted_labels = sorted(labels, key=lambda x: values[labels.index(x)])
    sorted_errors = sorted(errors, key=lambda x: values[errors.index(x)])

    # We build the plot
    x_pos = arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x_pos, sorted_values, yerr=sorted_errors, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_labels)
    ax.set_title('Hyperparameters importance')

    # We save the plot
    plt.savefig(os.path.join(path, HPS_IMPORTANCE_CHART))
    plt.close()


def compare_prediction_recordings(evaluations, split_index, recording_path=""):
    """Function that will plot a scatter plot showing the prediction of multiple experiments and the target value

    :param evaluations: List of strings representing the names of the evaluations to compare
    :param split_index: The index of the split we want to compare
    :param recording_path: the path to the recordings folder where we want to save the data
    """

    colors = ["blue", "red", "orange"]

    assert len(evaluations) >= 1, "at lest one evaluation must be specified"
    assert len(evaluations) <= 3, "maximum number of evaluations exceeded"

    # We create the paths to recoding files
    paths = [os.path.join(recording_path, evaluation, f"Split_{split_index}", RECORDS_FILE) for
             evaluation in evaluations]

    all_data = []

    # We get the data from the recordings
    for path in paths:
        # We read the record file of the first evaluation
        with open(path, "r") as read_file:
            all_data.append(json.load(read_file))

    comparison_possible = True
    ids = list(all_data[0][TEST_RESULTS].keys())

    # We check if the two evaluations are made on the same patients
    for i, data in enumerate(all_data):
        if i == 0:
            continue
        if len(data[TEST_RESULTS]) != len(all_data[0][TEST_RESULTS]):
            comparison_possible = False
            break
        id_to_compare = list(data[TEST_RESULTS].keys())

        for j, id in enumerate(id_to_compare):
            if id != ids[j]:
                comparison_possible = False
                break

    assert comparison_possible is True, "Different patients present in the given evaluations"

    target, ids, all_predictions = [], [], []

    # We gather the needed data from the recordings
    for i, data in enumerate(all_data):
        all_predictions.append([])
        for id, item in data[TEST_RESULTS].items():
            if i == 0:
                ids.append(id)
                target.append(float(item[TARGET]))
            all_predictions[i].append(float(item[PREDICTION]))

    # We sort all the predictions and the ids based on the target
    indexes = list(range(len(target)))
    indexes.sort(key=target.__getitem__)

    sorted_all_predictions = []
    for predictions in all_predictions:
        sorted_all_predictions.append([predictions[i] for i in indexes])
    sorted_target = [target[i] for i in indexes]
    sorted_ids = [ids[i] for i in indexes]

    # We set some parameters of the plot
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.rcParams['xtick.labelsize'] = 6

    # We create the scatter plot
    plt.scatter(sorted_ids, sorted_target, color='green', label="target")
    for i, predictions in enumerate(sorted_all_predictions):
        plt.scatter(sorted_ids, predictions, color=colors[i], label=evaluations[i])

    # We add the legend of the plot
    plt.legend()

    plt.title("Test set predictions and ground truth")

    # We save the plot
    plt.savefig(os.path.join(recording_path, evaluations[0], f"Split_{split_index}",
                             f"""comparison_{"_".join(evaluations)}.png"""))
    plt.close()
