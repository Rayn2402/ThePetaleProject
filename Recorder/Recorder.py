"""
Authors : Mehdi Mitiche

Files that contains the logic to save the results of the experiments

"""
import os
import pickle
import json
from torch.nn import Softmax
from numpy import std, min, max, mean, median


class Recorder:
    def __init__(self, evaluation_name, index):
        """
        Class that will be responsible of saving all the data about our experiments

        :param evaluation_name: String that represents the name of the evaluation
        :param index: The number of the split
        """
        folder_name = f"Split_{index}"

        # We create the folder where the information will be saved
        os.makedirs(os.path.join("Recordings/", evaluation_name, folder_name), exist_ok=True)

        self.path = os.path.join("Recordings/", evaluation_name, folder_name)
        self.data = {"name": evaluation_name, "index": index, "metrics": {}}

    def record_model(self, model):
        """
        Method to call to save a model with pickle
        """

        # We save the model with pickle
        filepath = os.path.join(self.path, "model.sav")
        pickle.dump(model, open(filepath, "wb"))

    def record_hyperparameters(self, hyperparameters):
        """
        Method to call to save the hyperparameters
        """

        # We save all the hyperparameters

        self.data["hyperparameters"] = [
            {key: round(hyperparameters[key], 6) if isinstance(hyperparameters[key], float) else hyperparameters[key]}
            for key in hyperparameters.keys()]

    def record_hyperparameters_importance(self, hyperparameter_importance):
        """
                Method to call to save the hyperparameter importance
                """

        # We save all the hyperparameter importance

        self.data["hyperparameter_importance"] = [
            {key: round(hyperparameter_importance[key], 6)} for key in hyperparameter_importance.keys()
        ]

    def record_scores(self, score, metric):
        """
        Method to call to save the scores of an experiments
        """

        # We save the score of the given metric
        self.data["metrics"][metric] = round(score, 6)

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
        """

        # We initialize the Softmax object
        softmax = Softmax(dim=1)
        predictions = softmax(predictions)

        # We save the predictions
        self.data["predictions"] = [{i: predictions[i].tolist()} for i in range(len(predictions))]


class RFRecorder(Recorder):
    """
        Class that will be responsible of saving all the data about our experiments with Random Forest
    """

    def __init__(self, evaluation_name, index):
        super().__init__(evaluation_name=evaluation_name, index=index)

    def record_predictions(self, predictions):
        """
        Method to call to save the predictions of a Random forestafter an experiments
        """

        # We save the predictions
        self.data["predictions"] = [{i: predictions[i]} for i in range(len(predictions))]


def get_evaluation_recap(evaluation_name):
    """
    Function that will create a JSON file containing the evaluation recap
    """
    assert os.path.exists(os.path.join("Recordings/", evaluation_name)), "Evaluation not found"
    path = os.path.join("Recordings/", evaluation_name)
    json_file = "records.json"
    folders = os.listdir(os.path.join(path))

    data = {"accuracy": {
        "values": [],
        "info":""
    },
    }

    for folder in folders:
        with open(os.path.join(f"{path}/{folder}/{json_file}"), "r") as read_file:
            split_data = json.load(read_file)
            data["accuracy"]["values"].append(split_data["metrics"]["ACCURACY"])

    values = data["accuracy"]["values"]
    data["accuracy"]["info"] = f"{mean(values)} +- {std(values)} [{median(values)},{min(values)}-{max(values)}]"

    with open(os.path.join(path, "general.json"), "w") as file:
        json.dump(data, file, indent=True)
