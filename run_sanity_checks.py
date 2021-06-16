"""
Runs all the sanity checks one after the other
"""


from os.path import join
from settings.paths import Paths
from subprocess import check_call


FIXED_CMD = ['python']
EVALUATION_CHECK = [join(Paths.SANITY_CHECKS, "evaluation_process.py")]
SAMPLING_CHECK = [join(Paths.SANITY_CHECKS, "sampling_distribution.py")]
TRAINING_CHECK = [join(Paths.SANITY_CHECKS, "training_process.py")]
TUNING_CHECK = [join(Paths.SANITY_CHECKS, "tuning_process.py")]

if __name__ == '__main__':

    # Run sanity checks
    for cmd in [SAMPLING_CHECK, TRAINING_CHECK, TUNING_CHECK, EVALUATION_CHECK]:
        check_call(FIXED_CMD + cmd)