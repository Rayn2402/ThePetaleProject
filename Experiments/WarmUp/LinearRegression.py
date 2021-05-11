"""
This file is used to store the experiment testing a linear regression model on the WarmUp dataset.
"""
from SQL.DataManager.Utils import PetaleDataManager
from Models.LinearModel import LinearRegressor
from Data.Sampling import get_warmup_sampler
from Utils.score_metrics import RegressionMetrics
from Recording.Recorder import RFRecorder, get_evaluation_recap, compare_prediction_recordings
from os.path import join


def execute_linear_regression_experience(k, regularization=False, lambda_values=[None]):
    manager = PetaleDataManager("mitm2902")

    evaluation_name = f"LinearRegression_{'r' if regularization is True else 'nr'}_k{k}"
    RECORDING_PATH = join("..", "..")

    # We create the warmup sampler to get the data
    warmup_sampler = get_warmup_sampler(dm=manager, outliers_ids=["P140", "P108"])
    data = warmup_sampler(k=k, valid_size=0, add_biases=True)



    features = ["B", "WEIGHT", "TDM6_HR_END", "TDM6_DIST", "DT", "AGE", "MVLPA"]

    for value in lambda_values:
        linear_regression_scores = []
        evaluation_name = f"{evaluation_name}_{value if value is not None else ''}"
        for i in range(k):
            # We create the linear regressor
            linear_regressor = LinearRegressor(input_size=7, regularization=regularization, lambda_value=value)

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

        print(linear_regression_scores)


execute_linear_regression_experience(k=10, regularization=False)
