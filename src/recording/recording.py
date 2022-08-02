"""
Filename: recording.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to define the Recorder class

Date of last modification : 2022/08/02
"""

import json
import matplotlib.pyplot as plt
import os
import pickle

from collections import Counter
from numpy import max, mean, median, min, std
from src.data.processing.datasets import MaskType
from src.models.abstract_models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.utils.visualization import visualize_importance, visualize_scaled_importance
from torch import tensor, save, zeros
from torch.nn import Module
from typing import Any, Dict, List, Optional, Union


class Recorder:
    """
    Recorder objects used save results of the experiments
    """
    # Constants
    ACCURACY: str = "accuracy"
    COEFFICIENT: str = "coefficient"
    DATA_INFO: str = "data_info"
    FEATURE_IMPORTANCE: str = "feature_importance"
    FEATURE_IMPORTANCE_CHART: str = 'feature_imp_recap.png'
    HPS_IMPORTANCE_CHART: str = 'hps_imp_recap.png'
    HYPERPARAMETERS: str = "hyperparameters"
    HYPERPARAMETER_IMPORTANCE: str = "hyperparameter_importance"
    INDEX: str = "index"
    INFO: str = "info"
    MEAN: str = "mean"
    METRICS: str = "metrics"
    NAME: str = "name"
    PREDICTION: str = "prediction"
    PREDS_N_TARGETS = "predictions_and_targets.png"
    RECORDS_FILE: str = "records.json"
    RESULTS: str = "results"
    S_FEATURE_IMPORTANCE_CHART: str = "scaled_feature_imp_recap.png"
    STD: str = "std"
    SUMMARY_FILE: str = "summary.json"
    TARGET: str = "target"
    TEST_METRICS: str = "test_metrics"
    TEST_RESULTS: str = "test_results"
    TRAIN_METRICS: str = "train_metrics"
    TRAIN_RESULTS: str = "train_results"
    VALID_METRICS: str = "valid_metrics"
    VALID_RESULTS: str = "valid_results"
    VALUES: str = "values"

    # Dictionary that associate the mask types to their proper section
    MASK_TO_SECTION: dict = {METRICS: {MaskType.TRAIN: TRAIN_METRICS,
                                       MaskType.TEST: TEST_METRICS,
                                       MaskType.VALID: VALID_METRICS},
                             RESULTS: {MaskType.TRAIN: TRAIN_RESULTS,
                                       MaskType.TEST: TEST_RESULTS,
                                       MaskType.VALID: VALID_RESULTS}
                       }

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
        self._data = {self.NAME: evaluation_name,
                      self.INDEX: index,
                      self.DATA_INFO: {},
                      self.HYPERPARAMETERS: {},
                      self.HYPERPARAMETER_IMPORTANCE: {},
                      self.FEATURE_IMPORTANCE: {},
                      self.TRAIN_METRICS: {},
                      self.TEST_METRICS: {},
                      self.VALID_METRICS: {},
                      self.COEFFICIENT: {},
                      self.TRAIN_RESULTS: {},
                      self.TEST_RESULTS: {},
                      self.VALID_RESULTS: {}}

        self._path = os.path.join(recordings_path, evaluation_name, f"Split_{index}")

        # We create the folder where the information will be saved
        os.makedirs(self._path, exist_ok=True)

    def generate_file(self) -> None:
        """
        Save the protected dictionary into a json file

        Returns: None
        """
        # We remove empty sections
        self._data = {k: v for k, v in self._data.items() if (k in [self.NAME, self.INDEX] or len(v) != 0)}

        # We save all the data collected in a json file
        filepath = os.path.join(self._path, self.RECORDS_FILE)
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
        self._data[self.COEFFICIENT][name] = value

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
        self._data[self.DATA_INFO][data_name] = data

    def record_features_importance(self, feature_importance: Dict[str, float]) -> None:
        """
        Saves the hyperparameters' importance in the protected dictionary

        Args:
            feature_importance: dictionary of features and their importance

        Returns: None
        """
        # We save all the hyperparameters importance
        for key in feature_importance.keys():
            self._data[self.FEATURE_IMPORTANCE][key] = round(feature_importance[key], 4)

    def record_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Saves the hyperparameters in the protected dictionary

        Args:
            hyperparameters: dictionary of hyperparameters and their value

        Returns: None
        """
        # We save all the hyperparameters
        for key in hyperparameters.keys():
            self._data[self.HYPERPARAMETERS][key] = round(hyperparameters[key], 6) if \
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
            self._data[self.HYPERPARAMETER_IMPORTANCE][key] = round(hyperparameter_importance[key], 4)

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
                      mask_type: str = MaskType.TRAIN) -> None:
        """
        Saves the score associated to a metric

        Args:
            score: float
            metric: name of the metric
            mask_type: train, test or valid

        Returns: None
        """
        # We find the proper section name
        section = Recorder.MASK_TO_SECTION[self.METRICS][mask_type]

        # We save the score of the given metric
        self._data[section][metric] = round(score, 6)

    def record_predictions(self,
                           ids: List[str],
                           predictions: tensor,
                           targets: Optional[tensor],
                           mask_type: str = MaskType.TRAIN) -> None:
        """
        Save the predictions of a given model for each patient ids

        Args:
            ids: patient/participant ids
            predictions: predicted class or regression value
            targets: target value
            mask_type: mask_type: train, test or valid

        Returns: None
        """
        # We find the proper section name
        section = Recorder.MASK_TO_SECTION[self.RESULTS][mask_type]

        # We save the predictions
        targets = targets if targets is not None else zeros(predictions.shape[0])
        if len(predictions.shape) == 0:
            for j, id_ in enumerate(ids):
                self._data[section][str(id_)] = {
                    self.PREDICTION: str(predictions[j].item()),
                    self.TARGET: str(targets[j].item())}
        else:
            for j, id_ in enumerate(ids):
                self._data[section][str(id_)] = {
                    self.PREDICTION: str(predictions[j].tolist()),
                    self.TARGET: str(targets[j].item())}

    def record_test_predictions(self,
                                ids: List[str],
                                predictions: tensor,
                                targets: tensor) -> None:
        """
        Records the test set's predictions

        Args:
            ids: list of patient/participant ids
            predictions: tensor with predicted targets
            targets: tensor with ground truth

        Returns: None
        """
        return self.record_predictions(ids, predictions, targets, mask_type=MaskType.TEST)

    def record_train_predictions(self,
                                 ids: List[str],
                                 predictions: tensor,
                                 targets: tensor) -> None:
        """
        Records the training set's predictions

        Args:
            ids: list of patient/participant ids
            predictions: tensor with predicted targets
            targets: tensor with ground truth

        Returns: None
        """
        return self.record_predictions(ids, predictions, targets)

    def record_valid_predictions(self,
                                 ids: List[str],
                                 predictions: tensor,
                                 targets: tensor) -> None:
        """
        Records the validation set's predictions

        Args:
            ids: list of patient/participant ids
            predictions: tensor with predicted targets
            targets: tensor with ground truth

        Returns: None
        """
        return self.record_predictions(ids, predictions, targets, mask_type=MaskType.VALID)


def get_evaluation_recap(evaluation_name: str,
                         recordings_path: str) -> None:
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
        Recorder.TRAIN_METRICS: {},
        Recorder.TEST_METRICS: {},
        Recorder.HYPERPARAMETER_IMPORTANCE: {},
        Recorder.FEATURE_IMPORTANCE: {},
        Recorder.HYPERPARAMETERS: {},
        Recorder.COEFFICIENT: {}
    }

    # Initialization of a list of key list that we can found within section of records dictionary
    key_lists = {}

    for folder in folders:

        # We open the json file containing the info of each split
        with open(os.path.join(path, folder, Recorder.RECORDS_FILE), "r") as read_file:
            split_data = json.load(read_file)

        # For each section and their respective key list
        for section in data.keys():
            if section in split_data.keys():

                # If the key list is not initialized yet..
                if key_lists.get(section) is None:

                    # Initialization of the key list
                    key_lists[section] = split_data[section].keys()

                    # Initialization of each individual key section in the dictionary
                    for key in key_lists[section]:
                        data[section][key] = {Recorder.VALUES: [], Recorder.INFO: ""}

                # We add values to each key associated to the current section
                for key in key_lists[section]:
                    data[section][key][Recorder.VALUES].append(split_data[section][key])

    # We remove empty sections
    data = {k: v for k, v in data.items() if len(v) != 0}

    # We add the info about the mean, the standard deviation, the median , the min, and the max
    set_info(data)

    # We save the json containing the summary of the records
    with open(os.path.join(path, Recorder.SUMMARY_FILE), "w") as file:
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
            values = data[section][key][Recorder.VALUES]

            if not (isinstance(values[0], str) or values[0] is None):
                mean_, std_ = round(mean(values), 4), round(std(values), 4)
                med_, min_, max_ = round(median(values), 4), round(min(values), 4), round(max(values), 4)
                data[section][key][Recorder.INFO] = f"{mean_} +- {std_} [{med_}; {min_}-{max_}]"
                data[section][key][Recorder.MEAN] = mean_
                data[section][key][Recorder.STD] = std_
            else:
                counts = Counter(data[section][key][Recorder.VALUES])
                data[section][key][Recorder.INFO] = str(dict(counts))


def plot_hps_importance_chart(evaluation_name: str,
                              recordings_path: str) -> None:
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
    with open(os.path.join(path, Recorder.SUMMARY_FILE), "r") as read_file:
        data = json.load(read_file)[Recorder.HYPERPARAMETER_IMPORTANCE]

    # We create the bar plot
    visualize_importance(data=data, figure_title='HPs importance',
                         filename=os.path.join(path, Recorder.HPS_IMPORTANCE_CHART))


def plot_feature_importance_charts(evaluation_name: str,
                                   recordings_path: str) -> None:
    """
    Creates a bar plots containing information about the mean and standard deviation
    of each feature's importance.

    Args:
        evaluation_name: name of the evaluation
        recordings_path: directory where containing the folders with the results of each split

    Returns: None

    """
    # We get the content of the json file
    path = os.path.join(recordings_path, evaluation_name)
    with open(os.path.join(path, Recorder.SUMMARY_FILE), "r") as read_file:
        data = json.load(read_file)[Recorder.FEATURE_IMPORTANCE]

    # We create the bar plots
    visualize_importance(data=data, figure_title='Features importance',
                         filename=os.path.join(path, Recorder.FEATURE_IMPORTANCE_CHART))
    visualize_scaled_importance(data=data, figure_title='Scaled feature importance',
                                filename=os.path.join(path, Recorder.S_FEATURE_IMPORTANCE_CHART))


def compare_prediction_recordings(evaluations: List[str],
                                  split_index: int,
                                  recording_path: str) -> None:
    """
    Creates a scatter plot showing the predictions of one or two
    experiments against the real labels.

    Args:
        evaluations: list of str representing the names of the evaluations to compare
        split_index: index of the split we want to compare
        recording_path: directory that stores the evaluations folder

    Returns: None
    """

    # We check that the number of evaluations provided is 2
    if not (1 <= len(evaluations) <= 2):
        raise ValueError("One or two evaluations must be specified")

    # We create the paths to recoding files
    paths = [os.path.join(recording_path, e, f"Split_{split_index}", Recorder.RECORDS_FILE) for e in evaluations]

    # We get the data from the recordings
    all_data = []  # List of dictionaries
    for path in paths:

        # We read the record file of the first evaluation
        with open(path, "r") as read_file:
            all_data.append(json.load(read_file))

    # We check if the two evaluations are made on the same patients
    comparison_possible = True
    first_experiment_ids = list(all_data[0][Recorder.TEST_RESULTS].keys())

    for i, data in enumerate(all_data[1:]):

        # We check the length of both predictions list
        if len(data[Recorder.TEST_RESULTS]) != len(all_data[0][Recorder.TEST_RESULTS]):
            comparison_possible = False
            break

        # We check ids in both list
        for j, id_ in enumerate(data[Recorder.TEST_RESULTS].keys()):
            if id_ != first_experiment_ids[j]:
                comparison_possible = False
                break

    if not comparison_possible:
        raise ValueError("Different patients are present in the given evaluations")

    targets, ids, all_predictions = [], [], []

    # We gather the needed data from the recordings
    for i, data in enumerate(all_data):

        # We add an empty list to store predictions
        all_predictions.append([])

        for id_, item in data[Recorder.TEST_RESULTS].items():

            # If we have not registered ids and targets yet
            if i == 0:
                ids.append(id_)
                targets.append(float(item[Recorder.TARGET]))

            all_predictions[i].append(float(item[Recorder.PREDICTION]))

    # We sort predictions and the ids based on their targets
    indexes = list(range(len(targets)))
    indexes.sort(key=lambda x: targets[x])
    all_predictions = [[predictions[i] for i in indexes] for predictions in all_predictions]
    targets = [targets[i] for i in indexes]
    ids = [ids[i] for i in indexes]

    # We set some parameters of the plot
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.rcParams["xtick.labelsize"] = 6

    # We create the scatter plot
    colors = ["blue", "orange"]
    plt.scatter(ids, targets, color="green", label="ground truth")
    for i, predictions in enumerate(all_predictions):
        plt.scatter(ids, predictions, color=colors[i], label=evaluations[i])

    # We add the legend and the title to the plot
    plt.legend()
    plt.title("Predictions and ground truth")

    # We save the plot
    plt.savefig(os.path.join(recording_path, evaluations[0], f"Split_{split_index}",
                             Recorder.PREDS_N_TARGETS))
    plt.close()
