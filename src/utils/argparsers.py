"""
Filename: argparsers.py

Author: Nicolas Raymond

Description: This file stores common argparser functions

Date of last modification: 2022/02/08
"""

import argparse


def apriori_argparser():
    """
    Creates a parser for the apriori experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python apriori_experiment.py',
                                     description="Runs the apriori algorithm on the different"
                                                 " split of a dataset")

    parser.add_argument('-min_sup', '--min_support', type=float, default=0.1,
                        help='Minimal support value (default = 0.1)')
    parser.add_argument('-min_conf', '--min_confidence', type=float, default=0.60,
                        help='Minimal confidence value (default = 0.60)')
    parser.add_argument('-min_lift', '--min_lift', type=float, default=1.20,
                        help='Minimal lift value (default = 1.20)')
    parser.add_argument('-max_length', '--max_length', type=int, default=1,
                        help='Max cardinality of item sets at the left side of rules (default = 1)')
    parser.add_argument('-nb_groups', '--nb_groups', type=int, default=2,
                        help='Number of quantiles considered to create the target groups if'
                             ' the target is continuous (default = 2)')

    arguments = parser.parse_args()

    # We show the arguments
    print_arguments(arguments)

    return arguments


def correct_and_smooth_parser():
    """
    Creates a parser for the correct and smooth experiment
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python label_propagation.py',
                                     description="Runs the correction and smoothing experiment")

    # Parameters selection
    parser.add_argument('-p', '--path', type=str,
                        help='Path of the folder from which to take predictions.'
                             '(Ex. records/experiments/warmup/enet...')
    parser.add_argument('-eval_name', '--evaluation_name', type=str,
                        help='Name of the evaluation (in order to name the folder with the results)')
    parser.add_argument('-nb_iter', '--nb_iter', type=int, default=1,
                        help='Number of correction adn smoothing iterations (default = 1)')
    parser.add_argument('-rc', '--r_correct', type=float, default=0.20,
                        help='Restart probability used in propagation algorithm for correction (default = 0.20)')
    parser.add_argument('-rs', '--r_smooth', type=float, default=0.80,
                        help='Restart probability used in propagation algorithm for smoothing (default = 0.80)')
    parser.add_argument('-gen1', '--genes_subgroup', default=False, action='store_true',
                        help='If true, only considers significant genes')
    parser.add_argument('-max_degree', '--max_degree', type=int, default=None,
                        help='Maximum degree of a nodes in the graph (default = None)')

    arguments = parser.parse_args()

    # We show the arguments
    print_arguments(arguments)

    return arguments


def print_arguments(arguments) -> None:
    """
    Prints the arguments of an argparser
    Args:
        arguments:

    Returns: None
    """
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

