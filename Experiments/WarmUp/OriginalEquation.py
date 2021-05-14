"""
This file is used to store the experiment testing the original equation on the WarmUp dataset.
"""
from SQL.DataManagement.Utils import PetaleDataManager
from Models.LinearModel import LinearRegressor
from Data.Sampling import get_warmup_sampler
from Utils.score_metrics import RegressionMetrics
from SQL.constants import *
from torch import from_numpy
import numpy as np
from Recording.Recorder import RFRecorder, get_evaluation_recap, compare_prediction_recordings
from os.path import join

EVALUATION_NAME = "OriginalEquation"
RECORDING_PATH = join("..", "..")


manager = PetaleDataManager("mitm2902")

# We create the warmup sampler to get the data
warmup_sampler = get_warmup_sampler(dm=manager, to_dataset=False)
data = warmup_sampler(k=10, valid_size=0)

original_equation_scores = []


# We create the function that will calculate the vo2 Peak value based on the original equation
def original_equation(item):
    return -0.236 * item[AGE] - 0.094 * item[WEIGHT] - 0.120 * item[TDM6_HR_END] + 0.067 * item[TDM6_DIST] + \
           0.065 * item[MVLPA] - 0.204 * item[DT] + 25.145


for i in range(10):
    original_equation_pred = []

    # We create the recorder
    recorder = RFRecorder(evaluation_name=EVALUATION_NAME, index=i, recordings_path=RECORDING_PATH)

    recorder.record_data_info("test_set", len(data[i]["test"]))

    # We get the predictions
    for index, row in data[i]["test"].X_cont.iterrows():
        original_equation_pred.append((original_equation(row)))

    # We save the predictions
    recorder.record_predictions(predictions=original_equation_pred, ids=data[i]["test"].IDs, target=data[i]["test"].y)

    # We calculate the score
    score = RegressionMetrics.mean_absolute_error(from_numpy(np.array(original_equation_pred)),
                                                  from_numpy(data[i]["test"].y))

    # We save the score
    recorder.record_scores(score=score, metric="mean_absolute_error")

    # We save the metric score
    original_equation_scores.append(score)

    # We generate the file containing the saved data
    recorder.generate_file()

    compare_prediction_recordings(evaluations=[EVALUATION_NAME], split_index=i, recording_path=RECORDING_PATH)


# We generate the evaluation recap
get_evaluation_recap(evaluation_name=EVALUATION_NAME, recordings_path=RECORDING_PATH)

print(original_equation_scores)
