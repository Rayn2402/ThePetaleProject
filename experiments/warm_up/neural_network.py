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
from src.models.nn_models import NNRegressor
from src.models.models_generation import NNModelGenerator
from src.training.evaluation import NNEvaluator
from src.data.processing.sampling import get_warmup_data, extract_masks
from src.data.processing.datasets import PetaleNNDataset
from settings.paths import Paths
from Experiments.warm_up.hps import NN_HPS
from src.data.extraction.constants import *
from typing import Callable, Dict, List


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 neural_network.py',
                                     description="Runs a neural network experiment")

    parser.add_argument('-k', '--nb_outer_splits', type=int, default=20,
                        help="Number of outer splits (default = [20])")
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=20,
                        help="Number of inner splits (default = [20])")
    parser.add_argument('-n', '--n_trials', type=int, default=100,
                        help="Number of trials (default = [100])")
    parser.add_argument('-s', '--seed', nargs="*", type=int, default=[SEED],
                        help=f"List of seeds (default = [{SEED}])")
    parser.add_argument('-u', '--user', type=str, default='rayn2402',
                        help="Valid username for petale database")
    parser.add_argument('-m', '--mvlpa', default=False, action='store_true',
                        help='Indicates if we want to consider MVLPA feature')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


def execute_neural_network_experiment(dataset: PetaleNNDataset, masks:Dict[int, Dict[str, List[int]]], n_trials: int,
                                      model_generator:NNModelGenerator, seed:List[int], evaluation_name:str)->Callable:
    """
    Function that executes a Neural Network experiments

     Args:
        dataset:  dataset with inputs and regression targets
        masks: dictionary with list of idx to use for training and testing
        n_trials: Number of trials to perform during the experiments
        model_generator: callable object used to generate a model according to a set of hyperparameters
        seed: random state used for reproducibility
        evaluation_name: name of the results file saved at the recordings_path


     Returns: None

     """

    # Initialization of the optimization metric
    metric = RootMeanSquaredError()

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = {"Root mean square error": RootMeanSquaredError(), "Mean absolute error": AbsoluteError()}

    # Creation of the evaluator
    nn_evaluator = NNEvaluator(model_generator=model_generator, dataset=dataset, masks=masks,
                               hps=NN_HPS, n_trials=n_trials, optimization_metric=metric,
                               evaluation_metrics=evaluation_metrics, max_epochs=100, early_stopping=True,
                               save_optimization_history=True, evaluation_name=evaluation_name, seed=seed)
    # Evaluation
    nn_evaluator.nested_cross_valid()


if __name__ == '__main__':
    # Arguments parsing
    args = argument_parser()

    # Arguments extraction
    n_trials = args.n_trials
    seeds = args.seed
    k = args.nb_outer_splits
    l = args.nb_inner_splits

    mvlpa = "MVLPA"

    # Generation of dataset
    data_manager = PetaleDataManager(args.user)
    df, target, cont_cols, _ = get_warmup_data(data_manager)
    if not args.mvlpa:
        df = df.drop([MVLPA], axis=1)
        cont_cols = [c for c in cont_cols if c != MVLPA]
        mvlpa=""


    # Creation of the dataset
    nn_dataset = PetaleNNDataset(df, VO2R_MAX, cont_cols, cat_cols=None)

    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "L0_masks.json"), k=k, l=l)

    # Creation of model generator
    nb_cont_cols = len(cont_cols)
    model_generator = NNModelGenerator(NNRegressor, nb_cont_cols, cat_sizes=None)

    # Experiments run
    for seed in seeds:

        # Naming of the evaluation
        evaluation_name = f"neural_network_k{k}_l{l}_n{n_trials}_s{seed}_{mvlpa}"

        # Execution of one experiment
        execute_neural_network_experiment(dataset=nn_dataset, masks=masks, seed=seed, n_trials=n_trials,
                                          model_generator=model_generator, evaluation_name=evaluation_name)