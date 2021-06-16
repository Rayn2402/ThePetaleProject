"""
This file is used to evaluate linear models in a regression context
"""

import argparse
import ray
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from itertools import product
from settings.paths import Paths
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from src.data.extraction.data_management import PetaleDataManager
from src.data.extraction.constants import SEED, MVLPA
from src.data.processing.datasets import PetaleRFDataset
from src.data.processing.sampling import TRAIN, TEST, get_warmup_data, RandomStratifiedSampler
from src.models.linear_models import LinearRegressor
from src.recording.recording import Recorder, get_evaluation_recap, compare_prediction_recordings
from src.utils.score_metrics import AbsoluteError
from torch import manual_seed, tensor, is_tensor
from typing import Callable, Dict, List


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 experiment.py [model]',
                                     description="Runs a polynomial regression experiment")

    parser.add_argument('-a', '--alpha', nargs="*", type=float, default=[0.0],
                        help="List of L1 penalty coefficient (default = [0.0])")
    parser.add_argument('-b', '--beta', nargs="*", type=float, default=[0.0],
                        help="List of L2 penalty coefficient (default = [0.0])")
    parser.add_argument('-d', '--degree', nargs="*", type=int, default=[1.0],
                        help="List of polynomial degrees (default = [1.0])")
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=20,
                        help="Number of outer splits (default = [20])")
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


def build_run(dataset: PetaleRFDataset, masks: Dict[int, Dict[str, List[int]]], mvlpa: bool = True) -> Callable:
    """
    Builds an experiment that can be used by ray
    Args:
        dataset: dataset with inputs and regression targets
        masks: dictionary with list of idx to use for training and testing
        mvlpa: True if we want to include moderate-to-vigorous leisure physical activity feature

    Returns: run_polynomial_regression function
    """

    @ray.remote
    def run_polynomial_regression(alpha: float, beta: float, degree: int) -> None:
        """
        Executes a polynomial regression experiment. Different models are used depending on the
        alpha parameter :

        - alpha = 0 -> Linear regression with analytical solution
        - alpha != 0 -> ElasticNet with coordinate descent

        Args:
            alpha: list of L1 penalty coefficient
            beta: list of L2 penalty coefficient
            degree: list of degrees to use for polynomial regression

        Returns: None

        """

        # We build the polynomial basis function
        poly_fit = PolynomialFeatures(degree)

        # We save the metric object
        metric = AbsoluteError()

        # We run training and testing for each masks
        for k, v in masks.items():

            # Masks extraction and dataset update
            train_mask, test_mask = v[TRAIN], v[TEST]
            dataset.update_masks(train_mask=train_mask, test_mask=test_mask)

            # Train and test data extraction
            x_train, y_train = dataset[train_mask]
            x_test, y_test = dataset[test_mask]
            x_train, x_test = x_train.to_numpy(), x_test.to_numpy()
            x_train, x_test = poly_fit.fit_transform(x_train), poly_fit.fit_transform(x_test)

            # We choose the model according to alpha parameter
            if alpha != 0:
                l1_ratio = alpha / (alpha + beta)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=10000)
                model_name = "elasticnet"
            else:
                model = LinearRegressor(input_size=x_train.shape[1], beta=beta)
                x_train, y_train, x_test, y_test = tensor(x_train), tensor(y_train), tensor(x_test), tensor(y_test)
                model_name = "linearreg"

            # Recorder initialization
            if mvlpa:
                evaluation_name = f"{model_name}_mvlpa_{seed}_{alpha}_{beta}_{degree}"
            else:
                evaluation_name = f"{model_name}_{seed}_{alpha}_{beta}_{degree}"

            recorder = Recorder(evaluation_name=evaluation_name,
                                index=k, recordings_path=Paths.EXPERIMENTS_RECORDS.value)

            # Model training
            model.fit(x_train, y_train)

            # Prediction calculations and recording
            predictions = model.predict(x_test)
            predictions = predictions if is_tensor(predictions) else tensor(predictions)
            recorder.record_predictions([dataset.ids[i] for i in test_mask], predictions, y_test)

            # Score calculation and recording
            score = metric(predictions, y_test)
            recorder.record_scores(score, "mean_absolute_error")

            # Generation of the file with the results
            recorder.generate_file()
            compare_prediction_recordings(evaluations=[evaluation_name], split_index=k,
                                          recording_path=Paths.EXPERIMENTS_RECORDS.value)
            get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS.value)

    return run_polynomial_regression


if __name__ == '__main__':

    # Arguments parsing
    args = argument_parser()

    # ray initialization
    ray.init()

    # Arguments extraction
    alphas = args.alpha
    betas = args.beta
    degrees = args.degree
    seeds = args.seed
    k = args.nb_outer_splits

    # Cartesian product of all inputs
    input_sets = product(alphas, betas, degrees)

    # Generation of dataset
    data_manager = PetaleDataManager(args.user)
    df, target, cont_cols, _ = get_warmup_data(data_manager)
    if not args.mvlpa:
        df = df.drop([MVLPA], axis=1)
        cont_cols = [c for c in cont_cols if c != MVLPA]
    dataset_ = PetaleRFDataset(df, target, cont_cols, cat_cols=None)

    # Experiments run
    for seed in seeds:
        manual_seed(seed)
        sampler = RandomStratifiedSampler(dataset_, n_out_split=k, n_in_split=0, valid_size=0)
        masks_ = sampler()
        run = build_run(dataset=dataset_, masks=masks_, mvlpa=args.mvlpa)
        futures = [run.remote(alpha=input_[0], beta=input_[1], degree=input_[2]) for i, input_ in enumerate(input_sets)]
        ray.get(futures)


