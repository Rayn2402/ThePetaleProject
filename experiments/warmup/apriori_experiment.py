"""
Filename: apriori_experiment.py

Authors: Nicolas Raymond

Description: This file is used to identify meaningful association
             rules between categorical values and VO2 max quantiles.

Date of last modification : 2022/02/08
"""

from os.path import dirname, realpath
from time import time

import sys


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.sampling import extract_masks, GeneChoice, get_warmup_data
    from src.utils.argparsers import apriori_argparser
    from src.utils.experiments import run_apriori_experiment

    # Arguments parsing
    args = apriori_argparser()

    # We save start time
    start = time()

    # We first extract data
    manager = PetaleDataManager()
    df, target, cont_cols, cat_cols = get_warmup_data(manager, genes=GeneChoice.ALL, sex=True)

    # Extraction of masks
    masks = extract_masks(Paths.WARMUP_MASK, k=10, l=0)

    # We run apriori experiment
    run_apriori_experiment(experiment_name='warmup',
                           df=df,
                           target=target,
                           cat_cols=cat_cols,
                           masks=masks,
                           arguments=args,
                           continuous_target=True)



