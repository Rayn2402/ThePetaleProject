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
    from src.data.processing.datasets import PetaleRFDataset
    from src.data.processing.sampling import get_learning_one_data, extract_masks, generate_multitask_labels
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.visualization import visualize_class_distribution

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # Extraction of the table
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True,
                                                    complications=[CARDIOMETABOLIC_COMPLICATIONS,
                                                                   BONE_COMPLICATIONS,
                                                                   NEUROCOGNITIVE_COMPLICATIONS,
                                                                   COMPLICATIONS])
    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS.value, "l1_masks.json"), k=1, l=0)

    # Creation of a dataset
    dataset = PetaleRFDataset(df, COMPLICATIONS, cont_cols, cat_cols)

    # Visualization of distribution
    labels_dict = {0: "No", 1: "Yes"}
    for k, v in masks.items():
        dataset.update_masks(train_mask=v["train"], valid_mask=v["valid"], test_mask=v["test"])
        if v != 'inner':
            for i in DATASET_TYPES:
                visualize_class_distribution(dataset.y[v[i]], labels_dict, title=f"{i}-{k}")
            for inner_k, inner_v in v['inner'].items():
                for j in DATASET_TYPES:
                    visualize_class_distribution(dataset.y[inner_v[j]], labels_dict, title=f"Inner {j}-{inner_k}")

    # Creation of multitask labels
    labels, labels_dict = generate_multitask_labels(df, [CARDIOMETABOLIC_COMPLICATIONS,
                                                         BONE_COMPLICATIONS,
                                                         NEUROCOGNITIVE_COMPLICATIONS])

    # Visualization of distribution for multitask labels
    for k, v in masks.items():
        if v != 'inner':
            for i in DATASET_TYPES:
                visualize_class_distribution(labels[v[i]], labels_dict, title=f"{i}-{k}")
            for inner_k, inner_v in v['inner'].items():
                for j in DATASET_TYPES:
                    visualize_class_distribution(labels[inner_v[j]], labels_dict, title=f"Inner {j}-{inner_k}")

