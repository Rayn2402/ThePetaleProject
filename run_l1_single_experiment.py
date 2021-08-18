"""
File use to run combination of l1 single experiments according to the health complication
"""

from os.path import join
from settings.paths import Paths
from subprocess import check_call
import argparse


def argument_parser():
    """
    This function defines a parser that enables user run l1 experiment according to health complication
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 full_experiment.py',
                                     description="Runs all the experiments associated to the l1 dataset"
                                                 " according to an health complication")

    # Health complication selection
    parser.add_argument('-comp', '--complication', type=str, default='bone',
                        choices=['bone', 'cardio', 'neuro', 'all'], help='Choice of health complication to predict')

    # Feature selection
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='True if we want to proceed to feature selection')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':

    # Arguments parsing
    args = argument_parser()

    # Extraction of file path
    FILE = str(join(Paths.L1_EXPERIMENT_SCRIPTS, "single_experiment.py"))

    # Creation of base commands
    cmd = ['python', FILE, '-base', '-comp', args.complication, '-han', '-logit', '-mlp',
           '-rf', '-xg', '-tab']

    # Creation of commands with feature selection
    if args.feature_selection:
        cmd += ['-f']

    # Run of experiments
    check_call(cmd)





