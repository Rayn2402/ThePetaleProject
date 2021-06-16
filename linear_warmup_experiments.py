"""
This file consists of the linear experiments made on the warmup dataset
"""

import time
import subprocess as sp
from os.path import join
from settings.paths import Paths


FILE = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS.value, "polynomial_regression.py"))
SEEDS = map(str, range(100, 106))
PENALTY_COEFFICIENTS = ['0.0', '0.005', '0.01', '0.05', '0.1', '0.25', '0.5', '1']
DEGREES = map(str, range(1, 4))
COMMANDS_1 = ['python3', FILE, '-a', *PENALTY_COEFFICIENTS, '-b', *PENALTY_COEFFICIENTS,
              '-d', *DEGREES, '-k', '20', '-s', *SEEDS]
COMMANDS_2 = COMMANDS_1 + ['-m']


if __name__ == '__main__':

    for cmd in [COMMANDS_1, COMMANDS_2]:

        # We run experiments
        start = time.time()
        p = sp.Popen(cmd)
        p.wait()

    print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))
