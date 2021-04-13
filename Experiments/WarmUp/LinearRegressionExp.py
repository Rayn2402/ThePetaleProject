"""
This file is used to store the experiment testing a linear regression model on the WarmUp dataset.
"""
from SQL.DataManager.Utils import PetaleDataManager
from Models.LinearModel import LinearRegressor
from Datasets.Sampling import WarmUpSampler
from Utils.score_metrics import RegressionMetrics
from Recorder.Recorder import Recorder, get_evaluation_recap
from os.path import join

manager = PetaleDataManager("mitm2902")

EVALUATION_NAME = "LinearRegression"
RECORDING_PATH = join("..", "..")


# We create the warmup sampler to get the data
warmup_sampler = WarmUpSampler(dm=manager)
data = warmup_sampler(k=10, valid_size=0, add_biases=True)

linear_regression_scores = []


for i in range(10):
    # We create the linear regressor
    linearRegressor = LinearRegressor(input_size=7)

    # We create the recorder
    recorder = Recorder(evaluation_name=EVALUATION_NAME, index=i, recordings_path=RECORDING_PATH)

    # We train the linear regressor
    linearRegressor.train(x=data[i]["train"].X_cont, y=data[i]["train"].y)

    # We make our predictions
    linear_regression_pred = linearRegressor.predict(x=data[i]["test"].X_cont)

    # We save the predictions
    recorder.record_predictions(linear_regression_pred.numpy().astype("float64"))

    # We calculate the score
    score = RegressionMetrics.mean_absolute_error(linear_regression_pred, data[i]["test"].y)

    # We save the score
    recorder.record_scores(score=score, metric="mean_absolute_error")

    # We save the metric score
    linear_regression_scores.append(score)

    # We generate the file containing the saved data
    recorder.generate_file()

# We generate the evaluation recap
get_evaluation_recap(evaluation_name=EVALUATION_NAME, recordings_path=RECORDING_PATH)

print(linear_regression_scores)
