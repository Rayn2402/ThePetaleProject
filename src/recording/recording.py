"""
Authors : Mehdi Mitiche
          Nicolas Raymond

Files that contains the logic to save the results of the experiments

"""
import json
import matplotlib.pyplot as plt
import os
import pickle

from numpy import std, min, max, mean, median, arange, argmax
from src.recording.constants import *
from sklearn.ensemble import RandomForestClassifier
from torch import tensor, save
from torch.nn import Module
from typing import Any, Dict, List, Union


class Recorder:
    """
    Recorder objects used save results of the experiments
    """
    def __init__(self, evaluation_name: str, index: int, recordings_path: str):
        """
        Sets protected attributes

        Args:
            evaluation_name: name of the evaluation
            index: index of the outer split
            recordings_path: path leading to where we want to save the results
        """

        # We store the protected attributes
        self._data = {NAME: evaluation_name, INDEX: index,
                      DATA_INFO: {}, HYPERPARAMETERS: {},
                      HYPERPARAMETER_IMPORTANCE: {}, METRICS: {},
                      COEFFICIENT: {}, RESULTS: {}}

        self._path = os.path.join(recordings_path, evaluation_name, f"Split_{index}")

        # We create the folder where the information will be saved
        os.makedirs(self._path, exist_ok=True)

    def generate_file(self) -> None:
        """
        Save the protected dictionary into a json file

        Returns: None

        """
        # We save all the data collected in a json file
        filepath = os.path.join(self._path, "records.json")
        with open(filepath, "w") as file:
            json.dump(self._data, file, indent=True)

    def record_coefficient(self, name: str, value: float) -> None:
        """
        Saves the value associated to a coefficient (used for linear regression)

        Args:
            name: name of the variable associated to the coefficient
            value: value of the coefficient

        Returns: None
        """
        self._data[COEFFICIENT][name] = value

    def record_data_info(self, data_name: str, data: Any) -> None:
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

    def record_model(self, model: Union[Module, RandomForestClassifier]) -> None:
        """
        Saves a model using pickle

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

    def record_scores(self, score: float, metric: str) -> None:
        """
        Saves the score associated to a metric

        Args:
            score: float
            metric: name of the metric

        Returns: None

        """
        # We save the score of the given metric
        self._data[METRICS][metric] = round(score, 6)

    def record_predictions(self, ids: List[str], predictions: tensor,
                           target: tensor) -> None:
        """
        Save the predictions of a given model for each patient ids

        Args:
            ids: patient/participant ids
            predictions: predicted class or regression value
            target: target value

        Returns: None

        """
        # We save the predictions
        if len(predictions.shape) == 0:
            for j, id_ in enumerate(ids):
                self._data[RESULTS][id_] = {
                    PREDICTION: predictions[j].item(),
                    TARGET: target[j].item()}
        else:
            for j, id_ in enumerate(ids):
                self._data[RESULTS][id_] = {
                    PREDICTION: predictions[j].tolist(),
                    TARGET: target[j].item()}


def get_evaluation_recap(evaluation_name, recordings_path):
    """
    Function that will create a JSON file containing the evaluation recap

    :param evaluation_name: The name of the evaluation
    :param recordings_path: the path to the recordings folder where we want to save the data
    """
    assert os.path.exists(os.path.join(recordings_path, evaluation_name)), "Evaluation not found"
    path = os.path.join(recordings_path, evaluation_name)
    json_file = "records.json"
    folders = next(os.walk(path))[1]
    data = {
        METRICS: {}
    }
    hyperparameter_importance_keys = None
    hyperparameters_keys = None
    metrics_keys = None
    coefficient_keys = None

    for folder in folders:
        # We open the json file containing the info of each split
        with open(os.path.join(path, folder, json_file), "r") as read_file:
            split_data = json.load(read_file)

        # We collect the info of the different metrics
        if metrics_keys is None:
            metrics_keys = split_data[METRICS].keys()
            for key in metrics_keys:
                data[METRICS][key] = {
                    VALUES: [],
                    INFO: ""
                }
        for key in metrics_keys:
            data[METRICS][key][VALUES].append(split_data[METRICS][key])

        # We collect the info of the different hyperparameter importance
        if HYPERPARAMETER_IMPORTANCE in split_data.keys():
            if HYPERPARAMETER_IMPORTANCE not in data.keys():
                data[HYPERPARAMETER_IMPORTANCE] = {}
            if hyperparameter_importance_keys is None:
                hyperparameter_importance_keys = split_data[HYPERPARAMETER_IMPORTANCE].keys()

                # We exclude the number of nodes from the hyperparameters importance (to be reviewed)
                hyperparameter_importance_keys = [key for key in hyperparameter_importance_keys if "n_units" not in key]
                for key in hyperparameter_importance_keys:
                    data[HYPERPARAMETER_IMPORTANCE][key] = {
                        VALUES: [],
                        INFO: ""
                    }
            for key in hyperparameter_importance_keys:
                data[HYPERPARAMETER_IMPORTANCE][key][VALUES].append(split_data[HYPERPARAMETER_IMPORTANCE][key])

        # We collect the info of the different hyperparameters
        if HYPERPARAMETERS in split_data.keys():
            if HYPERPARAMETERS not in data.keys():
                data[HYPERPARAMETERS] = {}
            if hyperparameters_keys is None:
                hyperparameters_keys = split_data[HYPERPARAMETERS].keys()

                # We exclude the layers from the hyperparameters importance (to be reviewed)
                hyperparameters_keys = [key for key in hyperparameters_keys if key not in ["layers", "activation"]]
                for key in hyperparameters_keys:
                    data[HYPERPARAMETERS][key] = {
                        VALUES: [],
                        INFO: ""
                    }
            for key in hyperparameters_keys:
                data[HYPERPARAMETERS][key][VALUES].append(split_data[HYPERPARAMETERS][key])

        # We collect the info of the different coefficient
        if COEFFICIENT in split_data.keys():
            if COEFFICIENT not in data.keys():
                data[COEFFICIENT] = {}
            if coefficient_keys is None:
                coefficient_keys = split_data[COEFFICIENT].keys()

                for key in coefficient_keys:
                    data[COEFFICIENT][key] = {
                        VALUES: [],
                        INFO: ""
                    }
            for key in coefficient_keys:
                data[COEFFICIENT][key][VALUES].append(split_data[COEFFICIENT][key])

    # We add the info about the mean, the standard deviation, the median , the min, and the max
    set_info(data)

    # We save the json containing the information about the evaluation in general
    with open(os.path.join(path, "general.json"), "w") as file:
        json.dump(data, file, indent=True)


def set_info(data):
    """
    Helper function that transforms the data to a specific format containing the mean, the standard deviation,
     the median , the min, and the max
    """
    for section in data.keys():
        for key in data[section].keys():
            data[section][key][
                INFO] = f"{round(mean(data[section][key][VALUES]), 4)} +- {round(std(data[section][key][VALUES]), 4)} " \
                        f"[{median(data[section][key][VALUES])}; {min(data[section][key][VALUES])}" \
                        f"-{max(data[section][key][VALUES])}]"
            data[section][key][MEAN] = mean(data[section][key][VALUES])
            data[section][key][STD] = std(data[section][key][VALUES])


def plot_hyperparameter_importance_chart(evaluation_name, recordings_path):
    """
    Function that will create a bar plot containing information about the mean and the standard deviation of each
    hyperparameter importance

    :param evaluation_name: String that represents the name of the evaluation
    :param recordings_path: the path to the recordings folder where we want to save the data

    """
    path = os.path.join(recordings_path, evaluation_name)
    json_file = "general.json"

    # We get the content of the json file
    with open(os.path.join(path, json_file), "r") as read_file:
        data = json.load(read_file)[HYPERPARAMETER_IMPORTANCE]

    # We initialize three lists for the values, the errors, and the labels
    values, errors, labels = [], [], []

    # We collect the data of each hyperparameter importance
    for key in data.keys():
        values.append(data[key][MEAN])
        errors.append(data[key][STD])
        labels.append(key)

    x_pos = arange(len(labels))

    # We sort the values
    sorted_values = sorted(values)
    sorted_labels = sorted(labels, key=lambda x: values[labels.index(x)])
    sorted_errors = sorted(errors, key=lambda x: values[errors.index(x)])

    # We build the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x_pos, sorted_values,
           yerr=sorted_errors,
           capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_labels)
    ax.set_title('Hyperparameters importance ')

    # We save the plot
    plt.savefig(os.path.join(path, 'hyperparameters_importance_recap.png'))
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
    paths = [os.path.join(recording_path, evaluation, f"Split_{split_index}", "records.json") for
             evaluation in evaluations]

    all_data = []

    # We get the data from the recordings
    for path in paths:
        # We read the record file of the first evaluation
        with open(path, "r") as read_file:
            all_data.append(json.load(read_file))

    comparaison_possible = True
    ids = list(all_data[0]["results"].keys())

    # We check if the two evaluations are made on the same patients
    for i, data in enumerate(all_data):
        if i == 0:
            continue
        if len(data["results"]) != len(all_data[0]["results"]):
            comparaison_possible = False
            break
        id_to_compare = list(data["results"].keys())

        for j,id in enumerate(id_to_compare):
            if id != ids[j]:
                comparaison_possible = False
                break

    assert comparaison_possible is True, "Different patients present in the given evaluations"

    target, ids, all_predictions = [], [], []

    # We gather the needed data from the recordings
    for i, data in enumerate(all_data):
        all_predictions.append([])
        for id, item in data["results"].items():
            if i == 0:
                ids.append((id))
                target.append(item["target"])
            all_predictions[i].append(argmax(item["prediction"]) if isinstance(item["prediction"], list)
                                      else item["prediction"])

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

    plt.title("visualization of the predictions and the ground truth")

    # We save the plot
    plt.savefig(os.path.join(recording_path, evaluations[0], f"Split_{split_index}",
                             f"""comparison_{"_".join(evaluations)}.png"""))
    plt.close()
