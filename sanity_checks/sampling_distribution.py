"""
This file is used to evaluate if the stratified sampling is correctly
done by the RandomStratifiedSampler class
"""

import sys
from os.path import join, realpath, dirname

DATASET_TYPES = ['train', 'valid', 'test']


if __name__ == '__main__':

    # STRATIFIED SAMPLING CHECK WITH LEARNING 01

    print(f"\nStratified sampling test...\n")

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.sampling import get_warmup_data, extract_masks
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.visualization import visualize_class_distribution

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # Extraction of the table
    df, _, cont_cols, cat_cols = get_warmup_data(manager, dummy=True)

    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "warmup_mask.json"), k=1, l=1)

    # Creation of a dataset
    dataset = PetaleDataset(df, WARMUP_DUMMY, cont_cols, cat_cols, classification=True)

    # Visualization of distribution
    for k, v in masks.items():
        dataset.update_masks(train_mask=v["train"], valid_mask=v["valid"], test_mask=v["test"])
        if v != 'inner':
            for i in DATASET_TYPES:
                _, y, _ = dataset[v[i]]
                visualize_class_distribution(y, WARMUP_DUMMY_DICT_NAME, title=f"{i}-{k}")
            for inner_k, inner_v in v['inner'].items():
                for j in DATASET_TYPES:
                    _, y, _ = dataset[inner_v[j]]
                    visualize_class_distribution(y, WARMUP_DUMMY_DICT_NAME, title=f"Inner {j}-{inner_k}")

