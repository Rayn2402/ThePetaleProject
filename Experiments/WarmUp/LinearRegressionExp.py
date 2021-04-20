"""
This file is used to store the experiment testing a linear regression model on the WarmUp dataset.
"""
from SQL.DataManager.Utils import PetaleDataManager
from Models.LinearModel import LinearRegressor
from Data.Sampling import get_warmup_sampler
from Utils.score_metrics import RegressionMetrics
from Recording.Recorder import RFRecorder, get_evaluation_recap, compare_prediction_recordings
from os.path import join

manager = PetaleDataManager("mitm2902")

EVALUATION_NAME = "LinearRegression"
RECORDING_PATH = join("..", "..")

# We create the warmup sampler to get the data
warmup_sampler = get_warmup_sampler(dm=manager)
data = warmup_sampler(k=10, valid_size=0, add_biases=True)

linear_regression_scores = []

features = ["B", "WEIGHT", "TDM6_HR_END", "TDM6_DIST", "DT", "AGE", "MVLPA"]

for i in range(10):
    # We create the linear regressor
    linearRegressor = LinearRegressor(input_size=7)

    # We create the recorder
    recorder = RFRecorder(evaluation_name=EVALUATION_NAME, index=i, recordings_path=RECORDING_PATH)

    recorder.record_data_info("train_set", len(data[i]["train"]))
    recorder.record_data_info("test_set", len(data[i]["test"]))

    # We train the linear regressor
    linearRegressor.train(x=data[i]["train"].X_cont, y=data[i]["train"].y)

    for j, feature in enumerate(features):
        recorder.record_coefficient(name=feature, value=linearRegressor.W[j].item())

    # We make our predictions
    linear_regression_pred = linearRegressor.predict(x=data[i]["test"].X_cont)

    # We save the predictions
    recorder.record_predictions(ids=data[i]["test"].IDs, predictions=linear_regression_pred.numpy().astype("float64"),
                                target=data[i]["test"].y.numpy().astype("float64"))

    # We calculate the score
    score = RegressionMetrics.mean_absolute_error(linear_regression_pred, data[i]["test"].y)

    # We save the score
    recorder.record_scores(score=score, metric="mean_absolute_error")

    # We save the metric score
    linear_regression_scores.append(score)

    # We generate the file containing the saved data
    recorder.generate_file()

    compare_prediction_recordings(evaluations=[EVALUATION_NAME], split_index=i, recording_path=RECORDING_PATH)

# We generate the evaluation recap
get_evaluation_recap(evaluation_name=EVALUATION_NAME, recordings_path=RECORDING_PATH)

print(linear_regression_scores)
