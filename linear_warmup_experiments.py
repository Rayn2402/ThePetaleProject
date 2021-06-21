"""
This file consists of the linear experiments made on the warmup dataset
"""

import time

from os.path import join
from settings.paths import Paths
from subprocess import check_call


FILE = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "polynomial_regression.py"))
DEGREES = map(str, range(1, 4))
COMMANDS = ['python3', FILE, '-d', *DEGREES, '-nos', '20', '-nis', '20', '-t', '150']

if __name__ == '__main__':

    # We run experiments
    start = time.time()
    check_call(COMMANDS)

    print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))
