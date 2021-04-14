"""
Authors : Mehdi Mitiche

Files that contains the logic to save the results of the experiments

"""
import os
import pickle
import json
from torch.nn import Softmax
from numpy import std, min, max, mean, median, arange
import matplotlib.pyplot as plt
from Recorder.constants import *


class Recorder:
    def __init__(self, evaluation_name, index):
        """
        Class that will be responsible of saving all the data about our experiments

        :param evaluation_name: String that represents the name of the evaluation
        :param index: The number of the split
        """
        folder_name = f"Split_{index}"

        # We create the folder where the information will be saved
        os.makedirs(os.path.join("Recordings", evaluation_name, folder_name), exist_ok=True)

        self.path = os.path.join("Recordings", evaluation_name, folder_name)
        self.data = {NAME: evaluation_name, INDEX: index, METRICS: {}, HYPERPARAMETERS: {},
                     HYPERPARAMETER_IMPORTANCE: {}}

    def record_model(self, model):
        """
        Method to call to save a model with pickle

        :param model: The model to save
        """

        # We save the model with pickle
        filepath = os.path.join(self.path, "model.sav")
        pickle.dump(model, open(filepath, "wb"))

    def record_hyperparameters(self, hyperparameters):
        """
        Method to call to save the hyperparameters

        :param hyperparameters: Python dictionary containing the hyperparameters to save
        """

        # We save all the hyperparameters
        for key in hyperparameters.keys():
            self.data[HYPERPARAMETERS][key] = round(hyperparameters[key], 6) if \
                isinstance(hyperparameters[key], float) else hyperparameters[key]

    def record_hyperparameters_importance(self, hyperparameter_importance):
        """
        Method to call to save the hyperparameter importance

        :param hyperparameter_importance: Python dictionary containing the hyperparameters importance to save
        """
        # We save all the hyperparameter importance
        for key in hyperparameter_importance.keys():
            self.data[HYPERPARAMETER_IMPORTANCE][key] = round(hyperparameter_importance[key], 4) if \
                isinstance(hyperparameter_importance[key], float) else hyperparameter_importance[key]

    def record_scores(self, score, metric):
        """
        Method to call to save the scores of an experiments

        :param score: The calculated score of a specific metric
        :param metric: The name of the metric to save
        """

        # We save the score of the given metric
        self.data[METRICS][metric] = round(score, 6)

    def generate_file(self):
        """
        Method to call to save the predictions of a model after an experiments
        """

        # We save all the data collected in a json file
        filepath = os.path.join(self.path, "records.json")
        with open(filepath, "w") as file:
            json.dump(self.data, file, indent=True)


class NNRecorder(Recorder):
    """
        Class that will be responsible of saving all the data about our experiments with Neural networks
    """

    def __init__(self, evaluation_name, index):
        super().__init__(evaluation_name=evaluation_name, index=index)

    def record_predictions(self, predictions):
        """
        Method to call to save the predictions of a neural network after an experiments

        :param predictions: The calculated predictions to save
        """

        # We initialize the Softmax object
        softmax = Softmax(dim=1)
        predictions = softmax(predictions)

        # We save the predictions
        self.data[PREDICTIONS] = [{i: predictions[i].tolist()} for i in range(len(predictions))]


class RFRecorder(Recorder):
    """
        Class that will be responsible of saving all the data about our experiments with Random Forest
    """

    def __init__(self, evaluation_name, index):
        super().__init__(evaluation_name=evaluation_name, index=index)

    def record_predictions(self, predictions):
        """
        Method to call to save the predictions of a Random forest after an experiments

        :param predictions: The calculated predictions to save
        """

        # We save the predictions
        self.data[PREDICTIONS] = [{i: predictions[i]} for i in range(len(predictions))]


def get_evaluation_recap(evaluation_name):
    """
    Function that will create a JSON file containing the evaluation recap

    :param evaluation_name: The name of the evaluation
    """
    assert os.path.exists(os.path.join("Recordings", evaluation_name)), "Evaluation not found"
    path = os.path.join("Recordings", evaluation_name)
    json_file = "records.json"
    folders = os.listdir(os.path.join(path))
    data = {
        METRICS: {
        },
        HYPERPARAMETER_IMPORTANCE: {

        }
    }

    hyperparameter_importance_keys = None
    metrics_keys = None
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
            data[section][key][INFO] = f"{round(mean(data[section][key][VALUES]),4)} +- {round(std(data[section][key][VALUES]),4)} " \
                                       f"[{median(data[section][key][VALUES])}; {min(data[section][key][VALUES])}" \
                                       f"-{max(data[section][key][VALUES])}]"
            data[section][key][MEAN] = mean(data[section][key][VALUES])
            data[section][key][STD] = std(data[section][key][VALUES])


def plot_hyperparameter_importance_chart(evaluation_name):
    """
    Function that will create a bar plot containing information about the mean and the standard deviation of each
    hyperparameter importance

    :param evaluation_name: String that represents the name of the evaluation

    """
    path = os.path.join("Recordings/", evaluation_name)
    json_file = "general.json"

    # We get the content of thte json file
    with open(os.path.join(f"{path}/{json_file}"), "r") as read_file:
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
