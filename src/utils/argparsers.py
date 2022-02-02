"""
Filename: argparsers.py

Author: Nicolas Raymond

Description: This file stores common argparser functions

Date of last modification: 2022/02/02
"""

import argparse


def apriori_argparser():
    """
    Creates a parser for the apriori experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python apriori_experiment.py',
                                     description="Runs the apriori algorithm on the different split of the"
                                                 "warmup dataset")

    parser.add_argument('-min_sup', '--min_support', type=float, default=0.1,
                        help='Minimal support value (default = 0.1)')
    parser.add_argument('-min_conf', '--min_confidence', type=float, default=0.60,
                        help='Minimal confidence value (default = 0.60)')
    parser.add_argument('-min_lift', '--min_lift', type=float, default=1.20,
                        help='Minimal lift value (default = 1.20)')
    parser.add_argument('-max_length', '--max_length', type=int, default=1,
                        help='Max cardinality of item sets at the left side of rules (default = 1)')
    parser.add_argument('-nb_groups', '--nb_groups', type=int, default=2,
                        help='Number quantiles considered to create V02 groups (default = 2)')

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

