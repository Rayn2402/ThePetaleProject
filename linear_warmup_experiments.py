"""
This file consists of the linear experiments made on the warmup dataset
"""

import time
import subprocess as sp
from os.path import join
from settings.paths import Paths
from src.data.extraction.constants import SEED

SEEDS = [str(SEED), '100', '102', '103', '104' ,'105']
PENALTY_COEFFICIENTS = ['0.0', '0.001', '0.005', '0.01', '0.05', '0.1']
DEGREES = ['1', '2', '3']
FILE = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS.value, "polynomial_regression.py"))
CMDS = ['python3', FILE, '-a', *PENALTY_COEFFICIENTS, '-b', *PENALTY_COEFFICIENTS,
        '-d', *DEGREES, '-k', '20']


if __name__ == '__main__':

    # We run experiments
    start = time.time()
    p = sp.Popen(CMDS)
    p.wait()

    print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))
