"""
Filename: fixed_models_comparisons.py

Author: Nicolas Raymond

Description: This file is a script used to run REF experiments using fixed
             hyperparameters.

Date of last modification: 2022/04/06
"""
import sys
from os.path import dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import ALL_CHROM_POS_REF, SIGNIFICANT_CHROM_POS_REF
    from src.data.processing.sampling import get_learning_two_data
    from src.utils.experiments import run_fixed_hps_regression_experiments

    # Run the experiments
    run_fixed_hps_regression_experiments(data_extraction_function=get_learning_two_data,
                                         mask_paths=[Paths.REF_MASK],
                                         experiment_id='ref',
                                         all_chrom_pos=ALL_CHROM_POS_REF,
                                         significant_chrom_pos=SIGNIFICANT_CHROM_POS_REF)
