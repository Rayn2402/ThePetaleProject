"""
File use to run all warmup experiments
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
                        help='True if we want to include genes if features')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='True if we want to apply automatic feature selection')
    parser.add_argument('-s', '--sex', default=False, action='store_true',
                        help='True if we want to include sex in features')
    parser.add_argument('-r_w', '--remove_walk_variables', default=False, action='store_true',
                        help='True if we want to remove six minutes walk test variables from baselines'
                             '(only applies if baselines are included')

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
    cmd = ['python', FILE, '-lin', '-han', '-mlp', '-rf', '-xg', '-tab']
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

    # Run of experiments
    check_call(cmd)





