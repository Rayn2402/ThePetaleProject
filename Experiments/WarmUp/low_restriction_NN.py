"""
This file is used to store the experiment testing a Neural Network on the WarmUp dataset.
"""
from os.path import join
from SQL.DataManager.Utils import PetaleDataManager
from Utils.score_metrics import RegressionMetrics
from Models.GeneralModels import NNRegressor
from Models.ModelGenerator import NNModelGenerator
from Evaluation.Evaluator import NNEvaluator
from json import load
from Data.Sampling import get_warmup_sampler

# We create the Petale Data Manager
manager = PetaleDataManager("mitm2902")

# We set the name of the evaluation
EVALUATION_NAME = "Low_restriction_NN"

# We set the path to the recordings folder
RECORDING_PATH = join("..", "..")

# We gather the hyperparameter to be optimized
with open(join("..", "..", "Hyperparameters", "low_restriction_hyperparameters.json"), "r") as read_file:
    HYPER_PARAMS = load(read_file)

# We create the warmup sampler to get the data
warmup_sampler = get_warmup_sampler(dm=manager)
data = warmup_sampler(k=10, valid_size=0)

# We define the evaluation metrics
evaluation_metrics = {
    "mean_absolute_error": RegressionMetrics.mean_absolute_error
}

# We create the mode generator
generator = NNModelGenerator(NNRegressor, num_cont_col=data[0]["train"].X_cont.shape[1])

# We create the Evaluator Object
NNEvaluator = NNEvaluator(model_generator=generator, sampler=warmup_sampler, k=10, l=10, hyper_params=HYPER_PARAMS,
                          optimization_metric=RegressionMetrics.mean_absolute_error,
                          evaluation_metrics=evaluation_metrics, n_trials=1000, seed=2019,
                          evaluation_name=EVALUATION_NAME, early_stopping_activated=True, max_epochs=300,
                          get_hyperparameters_importance=True, get_optimization_history=True,
                          get_parallel_coordinate=True, recordings_path=RECORDING_PATH)

# We perform the nested cross validation
score = NNEvaluator.nested_cross_valid()
