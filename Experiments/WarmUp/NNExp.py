"""
This file is used to store the experiment testing a Neural Network on the WarmUp dataset.
"""

from SQL.DataManager.Utils import PetaleDataManager
from Utils.score_metrics import RegressionMetrics
from Models.GeneralModels import NNRegressor
from Models.ModelGenerator import NNModelGenerator
from Evaluator.Evaluator import NNEvaluator
from json import load
from Datasets.Sampling import WarmUpSampler
from os import path

# We create the Petale Data Manager
manager = PetaleDataManager("mitm2902")

# We gather the hyperparameter to be optimized
with open(path.join("..", "..", "Hyperparameters", "hyper_params.json"), "r") as read_file:
    HYPER_PARAMS = load(read_file)

# We create the warmup sampler to get the data
warmup_sampler = WarmUpSampler(dm=manager)
data = warmup_sampler(k=10, valid_size=0)

# We define the evaluation metrics
evaluation_metrics = [
    {
        "name": "mean_absolute_error",
        "metric": RegressionMetrics.mean_absolute_error
    },
]

# We create the mode generator
generator = NNModelGenerator(NNRegressor, num_cont_col=data[0]["train"].X_cont.shape[1])

# We create the Evaluator Object
NNEvaluator = NNEvaluator(model_generator=generator, sampler=warmup_sampler, k=5, hyper_params=HYPER_PARAMS,
                          optimization_metric=RegressionMetrics.mean_absolute_error,
                          evaluation_metrics=evaluation_metrics, n_trials=300, seed=2019,
                          evaluation_name="NN_Regressor", early_stopping_activated=True,
                          get_hyperparameters_importance=True, get_optimization_history=True,
                          get_parallel_coordinate=True)

# We perform the nested cross validation
score = NNEvaluator.nested_cross_valid()

print("Neural Networks experiments completed")
print(score)