"""
Filename: full_experiment.py

Author: Nicolas Raymond

Description: This file is a script used to run obesity experiments using
             hyperparameter optimization.

Date of last modification: 2022/04/20
"""

import sys
from os.path import dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import SIGNIFICANT_CHROM_POS_OBESITY, ALL_CHROM_POS_OBESITY
    from src.data.processing.sampling import get_learning_one_data
    from src.utils.experiments import run_free_hps_regression_experiments

    # Run the experiments
    run_free_hps_regression_experiments(data_extraction_function=get_learning_one_data,
                                        mask_paths=[Paths.OBESITY_MASK, Paths.OBESITY_HOLDOUT_MASK],
                                        experiment_id='obesity',
                                        all_chrom_pos=ALL_CHROM_POS_OBESITY,
                                        significant_chrom_pos=SIGNIFICANT_CHROM_POS_OBESITY)
