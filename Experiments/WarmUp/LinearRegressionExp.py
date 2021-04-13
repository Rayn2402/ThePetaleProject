"""
This file is used to store the experiment testing a linear regression model on the WarmUp dataset.
"""
from SQL.DataManager.Utils import PetaleDataManager
from Models.LinearModel import LinearRegressor
from Datasets.Sampling import WarmUpSampler
from Utils.score_metrics import RegressionMetrics

manager = PetaleDataManager("mitm2902")

# We create the warmup sampler to get the data
warmup_sampler = WarmUpSampler(dm=manager)
data = warmup_sampler(k=10, valid_size=0, add_biases=True)

linear_regression_scores = []


for i in range(10):
    # We create the linear regressor
    linearRegressor = LinearRegressor(input_size=7)

    # We train the linear regressor
    linearRegressor.train(x=data[i]["train"].X_cont, y=data[i]["train"].y)

    # We make our predictions
    linear_regression_pred = linearRegressor.predict(x=data[i]["test"].X_cont)

    # We save the metric score
    linear_regression_scores.append(RegressionMetrics.mean_absolute_error(linear_regression_pred, data[i]["test"].y))

print(linear_regression_scores)
