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


def fixed_hps_lae_experiment_parser():
    """
    Creates a parser for the fixed hps lae experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python fixed_models_comparisons.py',
                                     description="Runs all the experiments associated a dataset,"
                                                 "using fixed hps")

    # Nb inner split and nb outer split selection
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=10,
                        help='Number of outer splits during the models evaluations')
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=10,
                        help='Number of inner splits during the models evaluations')

    # Features selection
    parser.add_argument('-gen1', '--genes_subgroup', default=False, action='store_true',
                        help='True if we want to include genes in features')
    parser.add_argument('-gen2', '--all_genes', default=False, action='store_true',
                        help='True if we want to include genes in features')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='True if we want to apply automatic feature selection')
    parser.add_argument('-imp', '--feature_imp_thresh', type=float, default=0.95,
                        help='Feature importance threshold to use if feature selection is activated')

    # Genes encoding parameter
    parser.add_argument('-share', '--embedding_sharing', default=False, action='store_true',
                        help='True if we want to use a single entity embedding layer for all genes'
                             ' (currently only applies with genomic signature creation')

    # Models selection
    parser.add_argument('-enet', '--enet', default=False, action='store_true',
                        help='True if we want to run enet experiment')
    parser.add_argument('-mlp', '--mlp', default=False, action='store_true',
                        help='True if we want to run mlp experiment')
    parser.add_argument('-rf', '--random_forest', default=False, action='store_true',
                        help='True if we want to run random forest experiment')
    parser.add_argument('-xg', '--xg_boost', default=False, action='store_true',
                        help='True if we want to run xgboost experiment')
    parser.add_argument('-gat', '--gat', default=False, action='store_true',
                        help='True if we want to run GraphAttentionNetwork experiment')
    parser.add_argument('-gge', '--gge', default=False, action='store_true',
                        help='True if we want to run GeneGraphEncoder with enet experiment')
    parser.add_argument('-ggae', '--ggae', default=False, action='store_true',
                        help='True if we want to run GeneGraphAttentionEncoder with enet experiment')

    # GAT graph construction parameters
    parser.add_argument('-w_sim', '--weighted_similarity', default=False, action='store_true',
                        help='True if we want to calculate patients similarities using weighted metrics')
    parser.add_argument('-cond_col', '--conditional_column', default=False, action='store_true',
                        help='True if we want to use the sex as conditional column in GAT construction')
    parser.add_argument('-deg', '--degree', nargs='*', type=str, default=[7],
                        help="Maximum number of neighbors for each node in the graph")

    # Gene encoding parameter
    parser.add_argument('-sign_size', '--signature_size', type=int, default=8,
                        help='Genomic signature size')

    # Self supervised learning experiments
    parser.add_argument('-ssl_ggae', '-ssl_ggae', default=False, action='store_true',
                        help='True if we want to run self supervised learning with the GeneGraphAttentionEncoder')
    parser.add_argument('-ssl_gge', '-ssl_gge', default=False, action='store_true',
                        help='True if we want to run self supervised learning with the GeneGraphEncoder')

    # Activation of sharpness-aware minimization
    parser.add_argument('-sam', '--enable_sam', default=False, action='store_true',
                        help='True if we want to use Sharpness-Aware Minimization Optimizer')


    # Usage of predictions from another experiment
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path leading to predictions of another model')

    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=1010710, help='Seed used during model evaluations')
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

