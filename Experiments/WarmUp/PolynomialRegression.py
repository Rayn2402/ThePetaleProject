"""
Author : Mehdi Mitiche

This file is used to store the procedure to test polynomial regression experiments on the WarmUp dataset.
"""
from SQL.DataManagement.Utils import PetaleDataManager
from Models.LinearModel import LinearRegressor
from Data.Sampling import get_warmup_sampler
from Utils.score_metrics import RegressionMetrics
from Recording.Recorder import RFRecorder, get_evaluation_recap, compare_prediction_recordings
from os.path import join
from sklearn.preprocessing import PolynomialFeatures
from torch import tensor, float32
from typing import List, Union


def execute_polynomial_regression(k: int, degree: int, lambda_values: List[float] = [0]):
    """
        Function that executes a linear regression experiments

        :param k: Number of outer splits
        :param degree: Number  representing the degree of the polynomial regression
        :param lambda_values: list of values of lambda to try in the case when we want to perform regularization
        """

    manager = PetaleDataManager("mitm2902")

    RECORDING_PATH = join(".")

    # We create the warmup sampler to get the data
    warmup_sampler = get_warmup_sampler(dm=manager)
    data = warmup_sampler(k=k, valid_size=0)

    for value in lambda_values:
        evaluation_name = f"PolynomialRegression_d{degree}_k{k}"
        evaluation_name = f"{evaluation_name}_r{value}"
        polynomial_regression_scores = []
        for i in range(k):
            # We transform the data
            polynomial_features = PolynomialFeatures(degree=degree)
            transformed_train_data = tensor(polynomial_features.fit_transform(data[i]["train"].X_cont), dtype=float32)
            transformed_test_data = tensor(polynomial_features.fit_transform(data[i]["test"].X_cont), dtype=float32)

            # We create the linear regressor
            linear_regressor = LinearRegressor(input_size=transformed_train_data.shape[1], lambda_value=value)

            # We create the recorder
            recorder = RFRecorder(evaluation_name=evaluation_name, index=i, recordings_path=RECORDING_PATH)

            recorder.record_data_info("train_set", len(transformed_train_data))
            recorder.record_data_info("test_set", len(transformed_test_data))

            # We train the linear regressor
            linear_regressor.train(x=transformed_train_data, y=data[i]["train"].y)

            # We make our predictions
            linear_regression_pred = linear_regressor.predict(x=transformed_test_data)

            # We save the predictions
            recorder.record_predictions(ids=data[i]["test"].IDs,
                                        predictions=linear_regression_pred.numpy().astype("float64"),
                                        target=data[i]["test"].y.numpy().astype("float64"))

            # We calculate the score
            score = RegressionMetrics.mean_absolute_error(linear_regression_pred, data[i]["test"].y)

            # We save the score
            recorder.record_scores(score=score, metric="mean_absolute_error")

            # We save the metric score
            polynomial_regression_scores.append(score)

            # We generate the file containing the saved data
            recorder.generate_file()

            compare_prediction_recordings(evaluations=[evaluation_name], split_index=i, recording_path=RECORDING_PATH)

        # We generate the evaluation recap
        get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=RECORDING_PATH)
