"""
This file is used to evaluate if the stratified sampling is correctly done by the Sampler class
"""

from SQL.NewTablesScripts.constants import FIT_LVLS_DICT
from SQL.DataManager.Utils import PetaleDataManager
from Utils.visualization import visualize_class_distribution
from Data.Sampling import get_learning_one_sampler

DATASET_TYPES = ['train', 'valid', 'test']


if __name__ == '__main__':

    # STRATIFIED SAMPLING TEST WITH LEARNING 01

    print(f"\nStratified sampling test...\n")

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")
    sampler = get_learning_one_sampler(dm=manager)

    # Sampling of train, valid and test sets
    datasets = sampler(k=1, l=1)

    # Visualization of distribution
    for k, v in datasets.items():
        if v != 'inner':
            for i in DATASET_TYPES:
                visualize_class_distribution(v[i].y, FIT_LVLS_DICT, title=f"{i}-{k}")
            for inner_k, inner_v in v['inner'].items():
                for j in DATASET_TYPES:
                    visualize_class_distribution(inner_v[j].y, FIT_LVLS_DICT, title=f"Inner {j}-{inner_k}")

