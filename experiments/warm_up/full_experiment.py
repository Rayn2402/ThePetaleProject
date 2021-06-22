"""
This file consists of all the experiments made on the warmup dataset
"""
from os.path import dirname, realpath, join
import sys

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import time
import subprocess as sp
from settings.paths import Paths


FILE_NN = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "neural_network.py"))
FILE_OE = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "original_equation.py"))
SEEDS = map(str, range(100, 106))
COMMANDS_NN_1 = ['python3', FILE_NN, '-k', '10', '-l', '20', '-n', '1000', '-s', *SEEDS, "-m"]
SEEDS = map(str, range(100, 106))
COMMANDS_OE = ['python3', FILE_OE, '-k', '10', '-s', *SEEDS]


if __name__ == '__main__':

    for cmd in [COMMANDS_NN_1, COMMANDS_OE]:
        # We run experiments
        start = time.time()
        p = sp.Popen(cmd)
        p.wait()

    print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))