"""
Filename: fixed_models_comparisons.py

Author: Nicolas Raymond

Description: This file is a script used to run obesity experiments using fixed
             hyperparameters.

Date of last modification: 2022/04/06
"""
import sys
from os.path import dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import SIGNIFICANT_CHROM_POS_OBESITY, OBESITY_SNPS
    from src.data.processing.sampling import get_obesity_data
    from src.utils.experiments import run_fixed_hps_regression_experiments

    # Run the experiments
    run_fixed_hps_regression_experiments(data_extraction_function=get_obesity_data,
                                         mask_paths=[Paths.OBESITY_MASK, Paths.OBESITY_HOLDOUT_MASK],
                                         experiment_id='obesity',
                                         all_chrom_pos=OBESITY_SNPS,
                                         significant_chrom_pos=SIGNIFICANT_CHROM_POS_OBESITY)
