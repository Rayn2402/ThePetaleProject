"""
This file is used to evaluate linear models in a regression context
"""

import argparse
import sys
from json import load
from os.path import dirname, realpath, join

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from settings.paths import Paths
from src.data.extraction.data_management import PetaleDataManager
from src.data.extraction.constants import SEED
from src.data.processing.datasets import PetaleLinearModelDataset
from src.data.processing.sampling import get_warmup_data, extract_masks, push_valid_to_test
from src.training.evaluation import ElasticNetEvaluator
from src.utils.score_metrics import AbsoluteError, RootMeanSquaredError


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 polynomial_regression.py [degree] [seed] [user]',
                                     description="Runs a polynomial regression experiment")

    parser.add_argument('-d', '--degree', nargs="*", type=int, default=[1.0],
                        help="List of polynomial degrees (default = [1.0])")
    parser.add_argument('-nis', '--nb_inner_splits', type=int, default=20,
                        help="Number of inner splits to use for hps tuning (default = 20)")
    parser.add_argument('-nos', '--nb_outer_splits', type=int, default=20,
                        help="Number of outer splits to use for model evaluation (default = 20)")
    parser.add_argument('-s', '--seed', type=int, default=SEED,
                        help=f"Seed for evaluator initialization (default = {SEED})")
    parser.add_argument('-t', '--nb_trials', type=int, default=100,
                        help="Number of hyperparameters sets sampled and evaluate"
                             " by Optuna for each inner split (default = 100)")
    parser.add_argument('-u', '--user', type=str, default='rayn2402',
                        help="Valid username for petale database")

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


if __name__ == '__main__':

    # Arguments parsing
    args = argument_parser()

    # Arguments extraction
    k, l = args.nb_outer_splits, args.nb_inner_splits

    # Masks extraction
    masks = extract_masks(join(Paths.MASKS, "l0_masks.json"), k, l)
    push_valid_to_test(masks)

    # Hyperparameters extraction
    with open(join(Paths.HYPERPARAMETERS, "elastic_net_warmup.json"), "r") as read_file:
        hps = load(read_file)

    # Metric choice
    opt_metric = RootMeanSquaredError()
    eval_metrics = {"MAE": AbsoluteError(), "RMSE": opt_metric}

    # Generation of dataset
    data_manager = PetaleDataManager(args.user)
    df, target, cont_cols, _ = get_warmup_data(data_manager)

    for deg in args.degree:

        # Creation of a dataset
        dataset = PetaleLinearModelDataset(df, target, cont_cols, polynomial_degree=deg)

        # Creation of an evaluator
        evaluator = ElasticNetEvaluator(dataset=dataset, masks=masks, hps=hps, n_trials=args.nb_trials,
                                        optimization_metric=opt_metric,
                                        evaluation_metrics=eval_metrics,
                                        seed=args.seed, evaluation_name=f"elastic_net_warmup_k{k}_l{l}_deg{deg}",
                                        save_hps_importance=True, save_parallel_coordinates=True,
                                        save_optimization_history=True)
        # Model Evaluation
        evaluator.nested_cross_valid()


