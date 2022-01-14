"""
Filename: run_warmup_experiment

Author: Nicolas Raymond

Description: Runs all the model comparisons on the warmup dataset using different set of features and parameters.

Date of last modification: 2021/12/14
"""

from os.path import join
from settings.paths import Paths
from subprocess import check_call
import argparse


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python run_warmup_experiment.py',
                                     description="Runs all the experiments associated to the warmup dataset")

    # Features selection
    parser.add_argument('-b', '--baselines', default=False, action='store_true',
                        help='True if we want to include variables from original equation')
    parser.add_argument('-gen', '--genes', default=False, action='store_true',
                        help='True if we want to include genes in features')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='True if we want to apply automatic feature selection')
    parser.add_argument('-s', '--sex', default=False, action='store_true',
                        help='True if we want to include sex in features')
    parser.add_argument('-r_w', '--remove_walk_variables', default=False, action='store_true',
                        help='True if we want to remove six minutes walk test variables from baselines'
                             '(only applies if baselines are included')

    # Genes encoding
    parser.add_argument('-gen_emb', '--genomic_embedding', default=False, action='store_true',
                        help='True if we want to use genomic signature generation for linear regression model')

    # Usage of predictions from another experiment
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path leading to predictions of another model, will only be used by HAN if specified')

    # Activation of sharpness-aware minimization
    parser.add_argument('-sam', '--enable_sam', default=False, action='store_true',
                        help='True if we want to use Sharpness-Aware Minimization Optimizer')

    # Activation of self supervised learning
    parser.add_argument('-pre_training', '--pre_training', default=False, action='store_true',
                        help='True if we want to apply pre self supervised training to model'
                             'where it is enabled. Currently available for ENET with genes encoding')

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

    # Extraction of file path
    FILE = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "full_experiment.py"))

    # Creation of commands
    cmd = ['python', FILE, '-lin', '-han', '-han_e', '-mlp', '-rf', '-xg', '-tab']
    if args.baselines:
        cmd.append('-b')
        if args.remove_walk_variables:
            cmd.append('-r_w')
    if args.genes:
        cmd.append('-gen')
    if args.feature_selection:
        cmd.append('-f')
    if args.sex:
        cmd.append('-s')
    if args.enable_sam:
        cmd.append('-sam')
    if args.genomic_embedding:
        cmd.append('-gen_emb')
    if args.pre_training:
        cmd.append('-pre_training')
    if args.path is not None:
        cmd += ['-p', args.path]

    # Run of experiments
    check_call(cmd)





