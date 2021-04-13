"""
This file is used to store the experiment testing a linear regression model on the WarmUp dataset.
"""
from SQL.DataManager.Utils import PetaleDataManager
from Models.LinearModel import LinearRegressor
from Datasets.Sampling import WarmUpSampler
from Utils.score_metrics import RegressionMetrics
from SQL.NewTablesScripts.constants import *
from torch import from_numpy
import numpy as np

manager = PetaleDataManager("mitm2902")

# We create the warmup sampler to get the data
warmup_sampler = WarmUpSampler(dm=manager, to_dataset=False)
data = warmup_sampler(k=10, valid_size=0)

original_equation_scores = []


# We create the function that will calculate the vo2 Peak value based on the origin al equation
def original_equation(item):
    return -0.236 * item[AGE] - 0.094 * item[WEIGHT] - 0.120 * item[TDM6_HR_END] + 0.067 * item[TDM6_DIST] + \
           0.065 * item[MVLPA] - 0.204 * item[DT] + 25.145


for i in range(10):
    original_equation_pred = []

    for index, row in data[i]["test"].X_cont.iterrows():
        original_equation_pred.append((original_equation(row)))

    # We save the metric score
    original_equation_scores.append(RegressionMetrics.mean_absolute_error(from_numpy(np.array(original_equation_pred)),
                                                                          from_numpy(data[0]["test"].y)))

print(original_equation_scores)
