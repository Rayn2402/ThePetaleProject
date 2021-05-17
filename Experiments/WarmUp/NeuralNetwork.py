"""
Author : Mehdi Mitiche

This file is used to store the procedure to test Neural Networks experiments on the WarmUp dataset.
"""
from os.path import join
from SQL.DataManagement.Utils import PetaleDataManager
from Utils.score_metrics import RegressionMetrics
from Models.GeneralModels import NNRegressor
from Models.ModelGenerator import NNModelGenerator
from Evaluation.Evaluator import NNEvaluator
from json import load
from Data.Sampling import get_warmup_sampler
from typing import List
import ray


def execute_neural_network_experiment(k: int, l: int, n_trials: int, hp_files_ids: List[int]):
    """
    Function that executes a Neural Network experiments

    :param k: Number of outer splits
    :param l: Number of inner splits
    :param n_trials: Number of trials to perform during the experiments
    :param hp_files_ids: List of ids of the hyperparameters file to test
    """

    assert len(hp_files_ids) > 0, "At least one hyperparameter id must be specified"

    # We create the Petale Data Manager
    manager = PetaleDataManager("mitm2902")

    RECORDING_PATH = join(".")

    # We create the warmup sampler to get the data
    warmup_sampler = get_warmup_sampler(dm=manager)
    data = warmup_sampler(k=k, valid_size=0)

    # We define the evaluation metrics
    evaluation_metrics = {
        "mean_absolute_error": RegressionMetrics.mean_absolute_error
    }

    # We create the mode generator
    generator = NNModelGenerator(NNRegressor, num_cont_col=data[0]["train"].X_cont.shape[1])

    for hp_id in hp_files_ids:
        evaluation_name = f"aNeuralNetwork_n{n_trials}_k{k}_l{l}"
        # We set the hyperparameter file id
        hp_id = str(hp_id)
        hp_id = f"{(3-len(hp_id))*'0'}{hp_id}"

        #We set the name of the evaluation
        evaluation_name = f"{evaluation_name}_hp{hp_id}"

        # We gather the hyperparameter to be optimized
        with open(join(RECORDING_PATH, "Hyperparameters", f"TEST_1_{hp_id}.json"), "r") as read_file:
            HYPER_PARAMS = load(read_file)

        # We create the Evaluator Object
        evaluator = NNEvaluator(model_generator=generator, sampler=warmup_sampler, k=k, l=l,
                                hyper_params=HYPER_PARAMS,
                                optimization_metric=RegressionMetrics.mean_absolute_error,
                                evaluation_metrics=evaluation_metrics, n_trials=n_trials, seed=2019,
                                evaluation_name=evaluation_name, early_stopping_activated=True, max_epochs=500,
                                get_hyperparameters_importance=True, get_optimization_history=True,
                                get_parallel_coordinate=True, recordings_path=RECORDING_PATH)

        # We perform the nested cross validation
        evaluator.nested_cross_valid()

        ray.shutdown()

