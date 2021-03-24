"""
Authors : Mehdi Mitiche

Files that contains the logic to save the results of the experiments

"""
import os
import pickle
import json


class Recorder:
    def __init__(self, evaluation_name, index):
        """
        Class that will be responsible of saving all the data about our experiments

        :param evaluation_name: String that represents the name of the evaluation
        :param index: The number of the split
        """
        folder_name = f"{evaluation_name}_{index}"

        # We create the folder where the information will be saved
        os.makedirs(os.path.join("Recordings/", folder_name), exist_ok=True)

        self.path = os.path.join("Recordings/", folder_name)
        self.data = {"name": evaluation_name, "index": index}

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
        self.data["hyperparameters"] = [{key: hyperparameters[key]} for key in hyperparameters.keys()]

    def record_predictions(self, predictions):
        """
        Method to call to save the predictions of a model after an experiments
        """

        # We save the predictions
        self.data["predictions"] = [{i: predictions[i].tolist()} for i in range(len(predictions))]

    def record_scores(self, score, metric):
        """
        Method to call to save the scores of an experiments
        """

        # We save the score of the given metric
        self.data[metric] = score

    def generate_file(self):
        """
        Method to call to save the predictions of a model after an experiments
        """

        # We save all the data collected in a json file
        filepath = os.path.join(self.path, "records.json")
        with open(filepath, "w") as file:
            json.dump(self.data, file)
