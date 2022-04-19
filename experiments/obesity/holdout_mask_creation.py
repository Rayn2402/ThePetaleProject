"""
Filename: holdout_mask_creation.py

Author: Nicolas Raymond

Description: Contains the procedure used to create the holdout mask for the obesity final test

Date of last modification: 2022/04/19
"""

import sys
from json import dump
from os.path import dirname, join, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.processing.sampling import GeneChoice, get_learning_one_data, RandomStratifiedSampler
    from src.data.processing.datasets import MaskType, PetaleDataset
    from src.data.extraction.constants import LEARNING_1_HOLDOUT, WARMUP_DUMMY
    from src.data.extraction.data_management import PetaleDataManager

    # Initialization of the manager
    m = PetaleDataManager()

    # Learning set extraction
    df, _, cont_cols, cat_cols = get_learning_one_data(data_manager=m,
                                                       baselines=True,
                                                       genes=GeneChoice.ALL,
                                                       dummy=True)

    # Temporary dataset creation
    cat_cols.remove(WARMUP_DUMMY)
    dts = PetaleDataset(df, WARMUP_DUMMY, cont_cols, cat_cols, classification=True)
    learning_size = len(dts)

    # Mask creation
    sampler = RandomStratifiedSampler(dts, n_out_split=1, n_in_split=10, random_state=1010710)
    mask = sampler()

    # Mask modification
    holdout_size = m.get_table(LEARNING_1_HOLDOUT).shape[0]
    mask[0][MaskType.TRAIN] += mask[0][MaskType.TEST]
    mask[0][MaskType.TEST] = list(range(learning_size, learning_size + holdout_size))

    # Mask saving
    with open(join(Paths.MASKS, "obesity_holdout_mask.json"), "w") as file:
        dump(mask, file, indent=True)



