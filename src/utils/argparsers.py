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


def obesity_experiment_parser():
    """
    Creates a parser for the obesity experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python [experiment file].py',
                                     description="Runs the experiments associated the obesity dataset,"
                                                 "using manually selected hyperparameters")

    # Nb of inner splits and nb of outer splits
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=10,
                        help='Number of outer splits during the models evaluations')
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=10,
                        help='Number of inner splits during the models evaluations')
    parser.add_argument('-holdout', '--holdout', default=False, action='store_true',
                        help='If true, includes the holdout set data')

    # Feature selection
    parser.add_argument('-class', '--classification', default=False, action='store_true',
                        help='If true, runs classification task instead of regression')
    parser.add_argument('-b', '--baselines', default=False, action='store_true',
                        help='If true, includes baselines in the features')
    parser.add_argument('-gen', '--genomics', default=False, action='store_true',
                        help='If true, includes all the genomic variables (SNPs)')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='If true, applies automatic feature selection')

    # Genes encoding parameter
    parser.add_argument('-share', '--embedding_sharing', default=False, action='store_true',
                        help='If true, uses a single entity embedding layer for all SNPs'
                             ' (only applied with genomic signature creation')

    # Models selection
    parser.add_argument('-enet', '--enet', default=False, action='store_true',
                        help='If true, runs enet experiment')
    parser.add_argument('-mlp', '--mlp', default=False, action='store_true',
                        help='If true, runs mlp experiment')
    parser.add_argument('-rf', '--random_forest', default=False, action='store_true',
                        help='If true, runs random forest experiment')
    parser.add_argument('-xg', '--xg_boost', default=False, action='store_true',
                        help='If true, runs xgboost experiment')
    parser.add_argument('-gat', '--gat', default=False, action='store_true',
                        help='If true, runs Graph Attention Network experiment')
    parser.add_argument('-gcn', '--gcn', default=False, action='store_true',
                        help='If true, runs Graph Convolutional Network experiment')
    parser.add_argument('-gge', '--gge', default=False, action='store_true',
                        help='If true, runs Gene Graph Encoder (with enet) experiment')
    parser.add_argument('-ggae', '--ggae', default=False, action='store_true',
                        help='If true, runs Gene GraphAttention Encoder (with enet) experiment')

    # Training parameters
    parser.add_argument('-epochs', '--epochs', type=int, default=500,
                        help='Maximal number of epochs during training')
    parser.add_argument('-patience', '--patience', type=int, default=50,
                        help='Number of epochs allowed without improvement (for early stopping)')

    # Graph construction parameters
    parser.add_argument('-w_sim', '--weighted_similarity', default=False, action='store_true',
                        help='If true, calculates patients similarities using weighted metrics')
    parser.add_argument('-cond_col', '--conditional_column', default=False, action='store_true',
                        help='If true, uses the sex as a conditional column in graph construction')
    parser.add_argument('-deg', '--degree', nargs='*', type=str, default=[7],
                        help="Maximum number of in-degrees for each node in the graph")

    # Gene encoding parameter
    parser.add_argument('-sign_size', '--signature_size', type=int, default=4,
                        help='Genomic signature size')

    # Self supervised learning experiments
    parser.add_argument('-ssl_ggae', '-ssl_ggae', default=False, action='store_true',
                        help='If true, runs self supervised learning with the Gene Graph Attention Encoder')
    parser.add_argument('-ssl_gge', '-ssl_gge', default=False, action='store_true',
                        help='If true, runs self supervised learning with the GeneGraphEncoder')

    # Activation of sharpness-aware minimization
    parser.add_argument('-rho', '--rho', type=float, default=0,
                        help='Rho parameter of Sharpness-Aware Minimization (SAM) Optimizer.'
                             'If >0, SAM is enabled')

    # Usage of predictions from another experiment
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path leading to predictions of another model')

    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=1010710,
                        help='Seed used during model evaluations')

    arguments = parser.parse_args()

    # Print arguments
    print_arguments(arguments)

    return arguments


def VO2_experiment_parser():
    """
    Creates a parser for VO2 peak prediction experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python [experiment file].py',
                                     description="Runs the experiments associated to the VO2 dataset")

    # Nb of inner splits and nb of outer splits
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=10,
                        help='Number of outer splits used during the models evaluations')
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=10,
                        help='Number of inner splits used during the models evaluations')
    parser.add_argument('-holdout', '--holdout', default=False, action='store_true',
                        help='If true, includes the holdout set data')

    # Feature selection
    parser.add_argument('-b', '--baselines', default=False, action='store_true',
                        help='If true, includes the variables from the original equation')
    parser.add_argument('-r_w', '--remove_walk_variables', default=False, action='store_true',
                        help='If true, removes the six-minute walk test variables from the baselines'
                             '(only effective if the baselines are included')
    parser.add_argument('-gen', '--genomics', default=False, action='store_true',
                        help='If true, includes all the genomic variables (SNPs)')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='If true, applies automatic feature selection')
    parser.add_argument('-s', '--sex', default=False, action='store_true',
                        help='If true, includes the sex in features')

    # Genes encoding parameter
    parser.add_argument('-share', '--embedding_sharing', default=False, action='store_true',
                        help='If true, uses a single entity embedding layer for all SNPs'
                             ' (only applied with genomic signature creation')

    # Models selection
    parser.add_argument('-enet', '--enet', default=False, action='store_true',
                        help='If true, runs enet experiment')
    parser.add_argument('-mlp', '--mlp', default=False, action='store_true',
                        help='If true, runs mlp experiment')
    parser.add_argument('-rf', '--random_forest', default=False, action='store_true',
                        help='If true, runs random forest experiment')
    parser.add_argument('-xg', '--xg_boost', default=False, action='store_true',
                        help='If true, runs xgboost experiment')
    parser.add_argument('-gat', '--gat', default=False, action='store_true',
                        help='If true, runs Graph Attention Network experiment')
    parser.add_argument('-gcn', '--gcn', default=False, action='store_true',
                        help='If true, runs Graph Convolutional Network experiment')
    parser.add_argument('-gge', '--gge', default=False, action='store_true',
                        help='If true, runs Gene Graph Encoder (with enet) experiment')
    parser.add_argument('-ggae', '--ggae', default=False, action='store_true',
                        help='If true, runs Gene GraphAttention Encoder (with enet) experiment')

    # Training parameters
    parser.add_argument('-epochs', '--epochs', type=int, default=500,
                        help='Maximal number of epochs during training')
    parser.add_argument('-patience', '--patience', type=int, default=50,
                        help='Number of epochs allowed without improvement (for early stopping)')

    # Graph construction parameters
    parser.add_argument('-w_sim', '--weighted_similarity', default=False, action='store_true',
                        help='If true, calculates patients similarities using weighted metrics')
    parser.add_argument('-cond_col', '--conditional_column', default=False, action='store_true',
                        help='If true, uses the sex as a conditional column in graph construction')
    parser.add_argument('-deg', '--degree', nargs='*', type=str, default=[7],
                        help="Maximum number of in-degrees for each node in the graph")

    # Gene encoding parameter
    parser.add_argument('-sign_size', '--signature_size', type=int, default=4,
                        help='Genomic signature size')

    # Self supervised learning experiments
    parser.add_argument('-ssl_ggae', '-ssl_ggae', default=False, action='store_true',
                        help='If true, runs self supervised learning with the Gene Graph Attention Encoder')
    parser.add_argument('-ssl_gge', '-ssl_gge', default=False, action='store_true',
                        help='If true, runs self supervised learning with the GeneGraphEncoder')

    # Activation of sharpness-aware minimization
    parser.add_argument('-rho', '--rho', type=float, default=0,
                        help='Rho parameter of Sharpness-Aware Minimization (SAM) Optimizer.'
                             'If >0, SAM is enabled')

    # Usage of predictions from another experiment
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path leading to predictions of another model')

    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=1010710,
                        help='Seed used during model evaluations')

    arguments = parser.parse_args()

    # Print arguments
    print_arguments(arguments)

    return arguments


def path_parser():
    """
    Provides an argparser that retrieves a path
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 file.py -p [path]',
                                     description="Stores a path")

    parser.add_argument('-p', '--path', type=str, help='Path of the experiment folder')
    arguments = parser.parse_args()
    print_arguments(arguments)

    return arguments


def print_arguments(arguments) -> None:
    """
    Prints the arguments of an argparser

    Args:
        arguments: arguments of an argparser

    Returns: None
    """
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

