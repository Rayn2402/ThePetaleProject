"""
Authors : Mehdi Mitiche

Files that contains the logic to save the results of the experiments

"""
import os
import pickle
import json
from torch.nn import Softmax
from numpy import std, min, max, mean, median, arange, argmax
import matplotlib.pyplot as plt
from Recording.constants import *


class Recorder:
    def __init__(self, evaluation_name, index, recordings_path):
        """
        Class that will be responsible of saving all the data about our experiments

        :param evaluation_name: String that represents the name of the evaluation
        :param index: The number of the split
        :param recordings_path: the path to the recordings folder where we want to save the data
        """
        folder_name = f"Split_{index}"

        # We create the folder where the information will be saved
        os.makedirs(os.path.join(recordings_path, "Recordings", evaluation_name, folder_name), exist_ok=True)

        self.path = os.path.join(recordings_path, "Recordings", evaluation_name, folder_name)
        self.data = {NAME: evaluation_name, INDEX: index}
        self.evaluation_name = evaluation_name

    def record_model(self, model):
        """
        Method to call to save a model with pickle

        :param model: The model to save
        """

        # We save the model with pickle
        filepath = os.path.join(self.path, "model.sav")
        pickle.dump(model, open(filepath, "wb"))

    def record_data_info(self, data_name, data):
        if DATA_INFO not in self.data.keys():
            self.data[DATA_INFO] = {}

        self.data[DATA_INFO][data_name] = data

    def record_hyperparameters(self, hyperparameters):
        """
        Method to call to save the hyperparameters

        :param hyperparameters: Python dictionary containing the hyperparameters to save
        """
        if HYPERPARAMETERS not in self.data.keys():
            self.data[HYPERPARAMETERS] = {}

        # We save all the hyperparameters
        for key in hyperparameters.keys():
            self.data[HYPERPARAMETERS][key] = round(hyperparameters[key], 6) if \
                isinstance(hyperparameters[key], float) else hyperparameters[key]

    def record_hyperparameters_importance(self, hyperparameter_importance):
        """
        Method to call to save the hyperparameter importance

        :param hyperparameter_importance: Python dictionary containing the hyperparameters importance to save
        """
        if HYPERPARAMETER_IMPORTANCE not in self.data.keys():
            self.data[HYPERPARAMETER_IMPORTANCE] = {}

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
        if METRICS not in self.data.keys():
            self.data[METRICS] = {}
        # We save the score of the given metric
        self.data[METRICS][metric] = round(score, 6)

    def record_coefficient(self, name, value):
        """
        Method to call to save the coefficient of a model

        :param name: The name of the coefficient
        :param value: The value of the coefficient
        """

        if COEFFICIENT not in self.data.keys():
            self.data[COEFFICIENT] = {}

        self.data[COEFFICIENT][name] = value

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

    def __init__(self, evaluation_name, index, recordings_path):
        super().__init__(evaluation_name=evaluation_name, index=index, recordings_path=recordings_path)

    def record_predictions(self, ids, predictions, target):
        """
        Method to call to save the predictions of a neural network after an experiments

        :param ids: The ids of the patients with the predicted data
        :param predictions: The calculated predictions to save
        :param target: The real values

        """

        if RESULTS not in self.data.keys():
            self.data[RESULTS] = {}

        # We save the predictions
        for i, id in enumerate(ids):
            self.data[RESULTS][id] = {
                PREDICTION: predictions[i].item() if len(predictions[i].shape) == 0 else predictions[i].tolist(),
                TARGET: target[i].item()}


class RFRecorder(Recorder):
    """
        Class that will be responsible of saving all the data about our experiments with Random Forest
    """

    def __init__(self, evaluation_name, index, recordings_path):
        super().__init__(evaluation_name=evaluation_name, index=index, recordings_path=recordings_path)

    def record_predictions(self, ids, predictions, target):
        """
        :param ids: The ids of the patients with the predicted data
        :param predictions: The calculated predictions to save
        :param target: The real values

        """
        if RESULTS not in self.data.keys():
            self.data[RESULTS] = {}

        # We save the predictions
        for i, id in enumerate(ids):
            self.data[RESULTS][id] = {PREDICTION: predictions[i], TARGET: target[i]}


def get_evaluation_recap(evaluation_name, recordings_path):
    """
    Function that will create a JSON file containing the evaluation recap

    :param evaluation_name: The name of the evaluation
    :param recordings_path: the path to the recordings folder where we want to save the data
    """
    assert os.path.exists(os.path.join(recordings_path, "Recordings", evaluation_name)), "Evaluation not found"
    path = os.path.join(recordings_path, "Recordings", evaluation_name)
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
    path = os.path.join(recordings_path, "Recordings", evaluation_name)
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


def compare_prediction_recordings(evaluations, split_index, recording_path=""):
    """Function that will plot a scatter plot showing the prediction of multiple experiments and the target value"""

    colors = ["blue", "red", "orange"]

    assert len(evaluations) >= 1, "at lest one evaluation must be specified"
    assert len(evaluations) <= 3, "maximum number of evaluations exceeded"

    # We create the paths to recoding files
    paths = [os.path.join(recording_path, "Recordings", evaluation, f"Split_{split_index}", "records.json") for
             evaluation in evaluations]

    all_data = []

    # We get the data from the recordings
    for path in paths:
        # We read the record file of the first evaluation
        with open(path, "r") as read_file:
            all_data.append(json.load(read_file))

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

    # We save the plot
    plt.savefig(os.path.join(recording_path, "Recordings", evaluations[0], f"Split_{split_index}",
                             f"""comparison_{"_".join(evaluations)}.png"""))
    plt.close()
