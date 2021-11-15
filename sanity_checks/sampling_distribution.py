"""
Filename: sampling_distribution.py

Author: Nicolas Raymond

Description: This file is used to evaluate if the stratified sampling is correctly
             done by the RandomStratifiedSampler class

Date of last modification: 2021/11/11
"""

import sys
from os.path import realpath, dirname

if __name__ == '__main__':

    # STRATIFIED SAMPLING CHECK WITH LEARNING 01
    print(f"\nStratified sampling test...\n")

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from settings.paths import Paths
    from src.data.processing.datasets import MaskType, PetaleDataset
    from src.data.processing.sampling import get_warmup_data, extract_masks
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.visualization import visualize_class_distribution

    # Initialization of DataManager and sampler
    manager = PetaleDataManager()

    # Extraction of the table
    df, _, cont_cols, cat_cols = get_warmup_data(manager, dummy=True)

    # Extraction of masks
    masks = extract_masks(Paths.WARMUP_MASK, k=1, l=1)

    # Creation of a dataset
    dataset = PetaleDataset(df, WARMUP_DUMMY, cont_cols, cat_cols, classification=True)

    # Visualization of distribution
    DATASET_TYPES = list(MaskType())
    for k, v in masks.items():
        dataset.update_masks(train_mask=v[MaskType.TRAIN], valid_mask=v[MaskType.VALID], test_mask=v[MaskType.TEST])
        if v != MaskType.INNER:
            for i in DATASET_TYPES:
                _, y, _ = dataset[v[i]]
                visualize_class_distribution(y, WARMUP_DUMMY_DICT_NAME, title=f"{i}-{k}")
            for inner_k, inner_v in v[MaskType.INNER].items():
                for j in DATASET_TYPES:
                    _, y, _ = dataset[inner_v[j]]
                    visualize_class_distribution(y, WARMUP_DUMMY_DICT_NAME, title=f"Inner {j}-{inner_k}")

