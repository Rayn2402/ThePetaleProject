"""


Author : Mehdi Mitiche

This file is used to store the procedure to test Logistic Regression experiments on the learning_01 dataset.
"""
import argparse
from os.path import dirname, realpath
import sys

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from os.path import join
from src.data.extraction.data_management import PetaleDataManager
from src.utils.score_metrics import Accuracy, CrossEntropyLoss, Sensitivity
from src.models.nn_models import NNClassifier
from src.models.models_generation import NNModelGenerator
from src.training.evaluation import NNEvaluator
from src.data.processing.sampling import get_learning_one_data, extract_masks
from src.data.processing.datasets import PetaleNNDataset
from settings.paths import Paths
from hps.hps import LOGISTIC_REGRESSION_HPS
from src.data.extraction.constants import *
from typing import Dict, List

SIGNIFICANT, ALL = "significant", "all"
COMPLICATIONS_CHOICES = [CARDIOMETABOLIC_COMPLICATIONS, BONE_COMPLICATIONS, NEUROCOGNITIVE_COMPLICATIONS, COMPLICATIONS]
GENES_CHOICES = [SIGNIFICANT, ALL]


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 logistic_regression.py',
                                     description="Runs a logistic regression  experiment")

    parser.add_argument('-nos', '--nb_outer_splits', type=int, default=20,
                        help="Number of outer splits (default = [20])")
    parser.add_argument('-nis', '--nb_inner_splits', type=int, default=20,
                        help="Number of inner splits (default = [20])")
    parser.add_argument('-t', '--nb_trials', type=int, default=100,
                        help="Number of trials (default = [100])")
    parser.add_argument('-s', '--seed', nargs="*", type=int, default=[SEED],
                        help=f"List of seeds (default = [{SEED}])")
    parser.add_argument('-u', '--user', type=str, default='rayn2402',
                        help="Valid username for petale database"),
    parser.add_argument('-c', '--complications_ids', nargs="*", type=int,
                        help="Ids of the complications we want to predict"),
    parser.add_argument('-g', '--genes', type=str,
                        help='Indicates the genes we want to consider ')
    parser.add_argument('-b', '--baselines', default=False, action='store_true',
                        help='Indicates if we want to baselines variables')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


def execute_logistic_regression_experiment(dataset: PetaleNNDataset, masks: Dict[int, Dict[str, List[int]]],
                                      n_trials: int, model_generator: NNModelGenerator, seed: int,
                                      evaluation_name:str) -> None:
    """
    Function that executes a Logistic Regression experiments

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
    metric = CrossEntropyLoss()

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = {"Sensitivity": Sensitivity(nb_classes=2), "Accuracy": Accuracy()}

    # Creation of the evaluator
    nn_evaluator = NNEvaluator(model_generator=model_generator, dataset=dataset, masks=masks,
                               hps=LOGISTIC_REGRESSION_HPS, n_trials=n_trials, optimization_metric=metric,
                               evaluation_metrics=evaluation_metrics, max_epochs=100, early_stopping=True,
                               save_optimization_history=True, evaluation_name=evaluation_name, seed=seed)
    # Evaluation
    nn_evaluator.nested_cross_valid()


if __name__ == '__main__':

    # Arguments parsing
    args = argument_parser()

    # Arguments extraction
    n_trials = args.nb_trials
    seeds = args.seed
    k = args.nb_outer_splits
    l = args.nb_inner_splits
    complications_ids = args.complications_ids
    genes = args.genes
    baselines = args.baselines

    if not isinstance(complications_ids, list):
        complications = [COMPLICATIONS_CHOICES[complications_ids]]
    else:
        complications = list(map(lambda x: COMPLICATIONS_CHOICES[x], complications_ids))

    if genes is not None:
        assert genes in GENES_CHOICES, f"genes value must be in {GENES_CHOICES}"

    b = "_b" if baselines else ""

    for c in complications:
        # Generation of dataset
        data_manager = PetaleDataManager(args.user)
        df, cont_cols, cat_cols = get_learning_one_data(data_manager=data_manager, baselines=baselines,
                                                         complications=[c], genes=genes)

        # Creation of the dataset
        nn_dataset = PetaleNNDataset(df=df, target=c, cont_cols=cont_cols, cat_cols=cat_cols)

        # Extraction of masks
        masks = extract_masks(join(Paths.MASKS, "L1_masks.json"), k=k, l=l)

        # Creation of model generator
        nb_cont_cols = len(cont_cols)
        cat_sizes = [len(v.items()) for v in nn_dataset.encodings.values()]
        model_generator = NNModelGenerator(NNClassifier, nb_cont_cols, cat_sizes, output_size=2)

        # Experiments run
        for seed in seeds:

            # Naming of the evaluation
            evaluation_name = f"l1_logistic_regression_k{k}_l{l}_n{n_trials}_s{seed}_c{c}_g{genes}{b}"

            # Execution of one experiment
            execute_logistic_regression_experiment(dataset=nn_dataset, masks=masks, seed=seed, n_trials=n_trials,
                                              model_generator=model_generator, evaluation_name=evaluation_name)
