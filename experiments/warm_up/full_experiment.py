"""
This file consists of all the experiments made on the warmup dataset
"""
from os.path import dirname, realpath, join
import sys

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import time
from subprocess import check_call
from settings.paths import Paths
from src.data.extraction.constants import SEED

FILE_NN = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "neural_network.py"))
FILE_OE = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "original_equation.py"))
FILE_PR = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "polynomial_regression.py"))

COMMANDS_NN = ['python3', FILE_NN, '-nos', '20', '-nis', '20', '-t', '1000', '-s', f'{SEED}', "-m"]

COMMANDS_OE = ['python3', FILE_OE, '-nos', '20', '-s', f'{SEED}']

DEGREES = map(str, range(1, 4))
COMMANDS_PR = ['python3', FILE_PR, '-d', *DEGREES, '-nos', '20', '-nis', '20', '-t', '1000', '-s', f'{SEED}']

if __name__ == '__main__':

    for cmd in [COMMANDS_NN, COMMANDS_OE, COMMANDS_PR]:
        # We run experiments
        start = time.time()
        check_call(cmd)

    print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))
