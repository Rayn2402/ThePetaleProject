"""
This file consists of all the experiments made on the l1 dataset
"""
from os.path import dirname, realpath, join
import sys

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import argparse
import time
from subprocess import check_call
from settings.paths import Paths
from src.data.extraction.constants import *

SIGNIFICANT, ALL = "significant", "all"
COMPLICATIONS_CHOICES = [CARDIOMETABOLIC_COMPLICATIONS, BONE_COMPLICATIONS, NEUROCOGNITIVE_COMPLICATIONS, COMPLICATIONS]
GENES_CHOICES = [SIGNIFICANT, ALL]


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 full_experiment.py',
                                     description="Runs all the experiments associated to the l1 dataset")

    parser.add_argument('-nn', '--neural_network', default=False, action='store_true',
                        help='Indicates if we want to run neural network experiments')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


COMMANDS_NN = []

FILE_NN = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "neural_network.py"))

COMPLICATIONS_IDS = map(str, range(0, 4))
COMMANDS_NN.append(['python3', FILE_NN, '-nos', '20', '-nis', '20', '-t', '1000', '-s', f'{SEED}', "-c",
                    *COMPLICATIONS_IDS])

COMPLICATIONS_IDS = map(str, range(0, 4))
COMMANDS_NN.append(['python3', FILE_NN, '-nos', '20', '-nis', '20', '-t', '1000', '-s', f'{SEED}', "-c",
                    *COMPLICATIONS_IDS, '-b'])

for gene_value in GENES_CHOICES:
    COMPLICATIONS_IDS = map(str, range(0, 4))
    COMMANDS_NN.append(['python3', FILE_NN, '-nos', '20', '-nis', '20', '-t', '1000', '-s', f'{SEED}', "-c",
                        *COMPLICATIONS_IDS, '-b', '-g', gene_value])

    COMPLICATIONS_IDS = map(str, range(0, 4))
    COMMANDS_NN.append(['python3', FILE_NN, '-nos', '20', '-nis', '20', '-t', '1000', '-s', f'{SEED}', "-c",
                        *COMPLICATIONS_IDS, '-g', gene_value])


if __name__ == '__main__':
    # Arguments parsing
    args = argument_parser()

    # Arguments extraction
    nn = args.neural_network

    # Preparation of the commands to execute
    commands = []
    if nn:
        commands.extend(COMMANDS_NN)

    if len(commands) == 0:
        print("Please choose one of the available options -nn, -oe, or -lin")
    else:
        start = time.time()
        for cmd in commands:
            # We run experiments
            check_call(cmd)

        print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))
