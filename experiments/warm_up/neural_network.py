"""


Author : Mehdi Mitiche

This file is used to store the procedure to test Neural Networks experiments on the WarmUp dataset.
"""
import argparse
from os.path import dirname, realpath
import sys

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from os.path import join
from src.data.extraction.data_management import PetaleDataManager
from src.utils.score_metrics import AbsoluteError, RootMeanSquaredError
from src.training.evaluation import NNEvaluator
from src.data.processing.sampling import get_warmup_data, extract_masks
from src.data.processing.datasets import PetaleNNDataset
from settings.paths import Paths
from hps.warmup_hps import NN_LOW_HPS, NN_HIGH_HPS, NN_ENET_HPS
from src.data.extraction.constants import *
from typing import Dict, List


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 neural_network.py',
                                     description="Runs a neural network experiment")
    parser.add_argument('-hps', '--hyperparameters', type=str, choices=['low', 'high', 'enet'],
                        help="Hyperparameters dictionary to use")
    parser.add_argument('-nos', '--nb_outer_splits', type=int, default=20,
                        help="Number of outer splits (default = [20])")
    parser.add_argument('-nis', '--nb_inner_splits', type=int, default=20,
                        help="Number of inner splits (default = [20])")
    parser.add_argument('-t', '--nb_trials', type=int, default=100,
                        help="Number of trials (default = [100])")
    parser.add_argument('-s', '--seed', nargs="*", type=int, default=[SEED],
                        help=f"List of seeds (default = [{SEED}])")
    parser.add_argument('-u', '--user', type=str, default='rayn2402',
                        help="Valid username for petale database")
    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


def execute_neural_network_experiment(dataset: PetaleNNDataset, masks: Dict[int, Dict[str, List[int]]],
                                      n_trials: int, seed: int, evaluation_name: str, hps: dict) -> None:
    """
    Function that executes a Neural Network experiments

     Args:
        dataset:  dataset with inputs and regression targets
        masks: dictionary with list of idx to use for training and testing
        n_trials: Number of trials to perform during the experiments
        seed: random state used for reproducibility
        evaluation_name: name of the results file saved at the recordings_path
        hps: hyperparameters dictionary


     Returns: None

     """

    # Initialization of the optimization metric
    metric = RootMeanSquaredError()

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = {"RMSE": metric, "MAE": AbsoluteError()}

    # Creation of the evaluator
    nn_evaluator = NNEvaluator(dataset=dataset, masks=masks,
                               hps=hps, n_trials=n_trials, optimization_metric=metric,
                               evaluation_metrics=evaluation_metrics, max_epochs=2000, early_stopping=True,
                               save_optimization_history=True, evaluation_name=evaluation_name, seed=seed)
    # Evaluation
    nn_evaluator.nested_cross_valid()


if __name__ == '__main__':

    # Arguments parsing
    args = argument_parser()

    # Arguments extraction
    n_trials = args.nb_trials
    seeds = args.seed
    hp_choice = args.hyperparameters
    k = args.nb_outer_splits
    l = args.nb_inner_splits

    # Generation of dataset
    data_manager = PetaleDataManager(args.user)
    df, target, cont_cols, _ = get_warmup_data(data_manager)

    # Creation of the dataset
    nn_dataset = PetaleNNDataset(df, target, cont_cols, classification=False)

    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "l0_masks.json"), k=k, l=l)

    # Extraction of hps
    if hp_choice == 'low':
        hyperparameters = NN_LOW_HPS
    elif hp_choice == 'high':
        hyperparameters = NN_HIGH_HPS
    else:
        hyperparameters = NN_ENET_HPS

    # Experiments run
    for seed in seeds:

        # Naming of the evaluation
        evaluation_name = f"neural_network_k{k}_l{l}_n{n_trials}_s{seed}_{hp_choice}"

        # Execution of one experiment
        execute_neural_network_experiment(dataset=nn_dataset, masks=masks, seed=seed, n_trials=n_trials,
                                          evaluation_name=evaluation_name, hps=hyperparameters)
