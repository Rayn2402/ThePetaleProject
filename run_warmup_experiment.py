"""
File use to run all warmup experiments
"""

from os.path import join
from settings.paths import Paths
from subprocess import check_call


if __name__ == '__main__':

    # Extraction of file path
    FILE = str(join(Paths.WARMUP_EXPERIMENTS_SCRIPTS, "full_experiment.py"))

    # Creation of commands
    cmd = ['python', FILE, '-lin']

    # Run of experiments
    check_call(cmd)





