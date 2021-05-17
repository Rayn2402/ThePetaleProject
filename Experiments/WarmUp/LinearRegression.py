"""
Author : Mehdi Mitiche

This file is used to store the procedure to test linear regression experiments on the WarmUp dataset.
"""
from SQL.DataManagement.Utils import PetaleDataManager
from Models.LinearModel import LinearRegressor
from Data.Sampling import get_warmup_sampler
from Utils.score_metrics import RegressionMetrics
from Recording.Recorder import RFRecorder, get_evaluation_recap, compare_prediction_recordings
from os.path import join
from typing import List, Union


def execute_linear_regression_experience(dm: PetaleDataManager,k: int, lambda_values: List[float] = [0]):
    """
    Function that executes a linear regression experiments

    :param dm: The Petale data manager
    :param k: Number of outer splits
    :param lambda_values: list of values of lambda to try in the case when we want to perform regularization
    """

    RECORDING_PATH = join(".")

    # We create the warmup sampler to get the data
    warmup_sampler = get_warmup_sampler(dm=dm)
    data = warmup_sampler(k=k, valid_size=0, add_biases=True)



    features = ["B", "WEIGHT", "TDM6_HR_END", "TDM6_DIST", "DT", "AGE", "MVLPA"]

    for value in lambda_values:
        linear_regression_scores = []
        evaluation_name = f"LinearRegression_k{k}"
        evaluation_name = f"{evaluation_name}_r{value}"
        for i in range(k):
            # We create the linear regressor
            linear_regressor = LinearRegressor(input_size=7, lambda_value=value)

            # We create the recorder
            recorder = RFRecorder(evaluation_name=evaluation_name, index=i, recordings_path=RECORDING_PATH)

            recorder.record_data_info("train_set", len(data[i]["train"]))
            recorder.record_data_info("test_set", len(data[i]["test"]))

            # We train the linear regressor
            linear_regressor.train(x=data[i]["train"].X_cont, y=data[i]["train"].y)

            for j, feature in enumerate(features):
                recorder.record_coefficient(name=feature, value=linear_regressor.W[j].item())

            # We make our predictions
            linear_regression_pred = linear_regressor.predict(x=data[i]["test"].X_cont)

            # We save the predictions
            recorder.record_predictions(ids=data[i]["test"].IDs,
                                        predictions=linear_regression_pred.numpy().astype("float64"),
                                        target=data[i]["test"].y.numpy().astype("float64"))

            # We calculate the score
            score = RegressionMetrics.mean_absolute_error(linear_regression_pred, data[i]["test"].y)

            # We save the score
            recorder.record_scores(score=score, metric="mean_absolute_error")

            # We save the metric score
            linear_regression_scores.append(score)

            # We generate the file containing the saved data
            recorder.generate_file()

            compare_prediction_recordings(evaluations=[evaluation_name], split_index=i, recording_path=RECORDING_PATH)

        # We generate the evaluation recap
        get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=RECORDING_PATH)

