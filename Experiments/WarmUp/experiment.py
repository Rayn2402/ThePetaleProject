"""
File:
    experiment.py

Authors:
    - Mehdi Mitiche

Description:
    Parsing Python command line arguments to run an experiment in the warmup dataset.
"""

import argparse
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
parent_parent = os.path.abspath(os.path.join(parent_dir_path, os.pardir))
sys.path.insert(0, parent_parent)
from Experiments.WarmUp.LinearRegression import execute_linear_regression_experience
from Experiments.WarmUp.PolynomialRegression import execute_polynomial_regression
from Experiments.WarmUp.NeuralNetwork import execute_neural_network_experiment
from SQL.DataManagement.Utils import PetaleDataManager

LINEAR_REGRESSION = "linear_regression"
POLYNOMIAL_REGRESSION = "polynomial_regression"
NEURAL_NETWORK = "neural_network"


def argument_parser():
    """
    This function defines a parser to enable user to easily experiment different models
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 experiment.py [model]',
                                     description="This program enables user to train different "
                                                 "models of regression to predict the  VO2 max "
                                                 "value.")

    parser.add_argument('-m', '--model', type=str,
                        choices=[LINEAR_REGRESSION, POLYNOMIAL_REGRESSION, NEURAL_NETWORK],
                        help=f"Type of the model to train ({LINEAR_REGRESSION}, {POLYNOMIAL_REGRESSION},"
                             f" or {NEURAL_NETWORK})")
    parser.add_argument('-k', '--k_splits', type=int, default=10,
                        help="Number of outer splits to be used in the experiments")

    parser.add_argument('-l', '--l_splits', type=int, default=10,
                        help='Number of inner splits to be used in the experiments '
                             'labeled in each class by the Expert')
    parser.add_argument('-d', '--degree', type=int, default=2,
                        help='Degree of the polynomial regression')
    parser.add_argument('-hp', '--hyperparameter_ids', type=int, default=1,
                        help='The id of the hyperparameters files to be used in the case of neural networks ')
    parser.add_argument('-t', '--trials', type=int, default=100,
                        help='Number of trials made in the optimization for the cas of neural networks')
    parser.add_argument('-y', '--lambda_values', type=int, default=0,
                        help='values of lambda for the regularization in the case of the linear regression'
                             'and the polynomial regression')
    parser.add_argument('-u', '--username', type=str, default="rayn2402",
                        help='username of the petale database')

    args = parser.parse_args()

    return args


def main():
    # We parse arguments
    args = argument_parser()

    # We extract arguments
    username = args.username
    model = args.model
    k = args.k_splits
    l = args.l_splits
    degree = args.degree
    hyperparameter_ids = args.hyperparameter_ids
    trials = args.trials
    lambda_values = args.lambda_values

    dm = PetaleDataManager(user=username)

    # We transform the args expected to be of type list
    if isinstance(lambda_values, int):
        lambda_values = [lambda_values]

    if isinstance(hyperparameter_ids, int):
        hyperparameter_ids = [hyperparameter_ids]

    # We call appropriate model
    if model == LINEAR_REGRESSION:
        execute_linear_regression_experience(dm=dm, k=k, lambda_values=lambda_values)
    elif model == POLYNOMIAL_REGRESSION:
        execute_polynomial_regression(dm=dm, k=k, degree=degree, lambda_values=lambda_values)
    elif model == NEURAL_NETWORK:
        execute_neural_network_experiment(dm=dm, k=k, l=l, n_trials=trials, hp_files_ids=hyperparameter_ids)


if __name__ == "__main__":
    main()
