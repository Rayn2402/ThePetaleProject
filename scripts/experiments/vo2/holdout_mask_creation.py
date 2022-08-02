"""
Filename: holdout_mask_creation.py

Author: Nicolas Raymond

Description: Contains the procedure used to create the holdout mask
             for the VO2 peak prediction final test

Date of last modification: 2022/08/02
"""

import sys
from json import dump
from os.path import dirname, join, realpath
from pandas import read_csv

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
    from settings.paths import Paths
    from src.data.processing.sampling import get_VO2_data, RandomStratifiedSampler
    from src.data.processing.datasets import MaskType, PetaleDataset
    from src.data.extraction.constants import VO2_HOLDOUT_SET, DUMMY
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.argparsers import data_source_parser

    # Arguments parsing
    args = data_source_parser()

    # Initialization of the manager and setting of the holdout size
    if not args.from_csv:
        m = PetaleDataManager()
        holdout_size = m.get_table(VO2_HOLDOUT_SET).shape[0]

    else:
        m = None
        holdout_size = read_csv(Paths.VO2_HOLDOUT_SET_CSV).shape[0]

    # Learning set extraction
    df, _, cont_cols, cat_cols = get_VO2_data(data_manager=m,
                                              baselines=True,
                                              genomics=True,
                                              sex=True,
                                              dummy=True)

    # Temporary dataset creation
    cat_cols.remove(DUMMY)
    dts = PetaleDataset(df, DUMMY, cont_cols, cat_cols, classification=True)
    learning_size = len(dts)

    # Mask creation
    sampler = RandomStratifiedSampler(dts, n_out_split=1, n_in_split=10, random_state=1010710)
    mask = sampler()

    # Mask modification
    mask[0][MaskType.TRAIN] += mask[0][MaskType.TEST]
    mask[0][MaskType.TEST] = list(range(learning_size, learning_size + holdout_size))

    # Mask saving
    with open(join(Paths.MASKS, "warmup_holdout_mask.json"), "w") as file:
        dump(mask, file, indent=True)



