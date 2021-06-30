"""
This file consists of all the experiments made on the warmup dataset
"""
from os.path import dirname, realpath, join
import sys

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import argparse
import time
from subprocess import check_call
from settings.paths import Paths
from src.data.extraction.constants import SEED


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 full_experiment.py',
                                     description="Runs all the experiments associated to the warmup dataset")

    parser.add_argument('-nn_low', '--neural_network_low', default=False, action='store_true',
                        help='Indicates if we want to run neural network experiments with low'
                             ' restriction hyperparameters')
    parser.add_argument('-nn_high', '--neural_network_high', default=False, action='store_true',
                        help='Indicates if we want to run neural network experiments with high'
                             ' restriction hyperparameters')
    parser.add_argument('-nn_enet', '--neural_network_enet', default=False, action='store_true',
                        help='Indicates if we want to run neural network experiments with enet'
                             ' hyperparameters')
    parser.add_argument('-oe', '--original_equation', default=False, action='store_true',
                        help='Indicates if we want to run original equation experiments')
    parser.add_argument('-lin', '--polynomial_regression', default=False, action='store_true',
                        help='Indicates if we want to run polynomial regression experiments')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


FILE_NN = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "neural_network.py"))
FILE_OE = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "original_equation.py"))
FILE_PR = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "polynomial_regression.py"))

COMMANDS_NN = ['python3', FILE_NN, '-nos', '20', '-nis', '20', '-t', '1000', '-s', f'{SEED}']
COMMANDS_NN_LOW = COMMANDS_NN + ['-hps', 'low']
COMMANDS_NN_HIGH = COMMANDS_NN + ['-hps', 'high']
COMMANDS_NN_ENET = COMMANDS_NN + ['-hps', 'enet']
COMMANDS_OE = ['python3', FILE_OE, '-nos', '20']

DEGREES = map(str, range(1, 4))
COMMANDS_PR = ['python3', FILE_PR, '-d', *DEGREES, '-nos', '20', '-nis', '20', '-t', '1000', '-s', f'{SEED}']

if __name__ == '__main__':
    # Arguments parsing
    args = argument_parser()

    # Arguments extraction
    nn_low = args.neural_network_low
    nn_high = args.neural_network_high
    nn_enet = args.neural_network_enet
    oe = args.original_equation
    lin = args.polynomial_regression

    # Preparation of the commands to execute
    commands = []
    if nn_low:
        commands.append(COMMANDS_NN_LOW)

    if nn_high:
        commands.append(COMMANDS_NN_HIGH)

    if nn_enet:
        commands.append(COMMANDS_NN_ENET)

    if oe:
        commands.append(COMMANDS_OE)

    if lin:
        commands.append(COMMANDS_PR)
    if len(commands) == 0:
        print("Please choose one of the available options -nn_low, -nn_enet, -oe, or -lin")
    else:
        start = time.time()
        for cmd in commands:

            # We run experiments
            check_call(cmd)

        print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))
