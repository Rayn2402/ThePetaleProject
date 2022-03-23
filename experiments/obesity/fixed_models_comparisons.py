"""
Filename: fixed_models_comparisons.py

Author: Nicolas Raymond

Description: This file is a script used to run obesity experiments using fixed
             hyperparameters.

Date of last modification: 2022/03/23
"""
import sys
from os.path import dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.processing.sampling import get_learning_one_data
    from src.utils.experiments import run_fixed_hps_regression_experiments

    # Run the experiments
    run_fixed_hps_regression_experiments(data_extraction_function=get_learning_one_data,
                                         mask_path=Paths.OBESITY_MASK,
                                         experiment_id='obesity')
