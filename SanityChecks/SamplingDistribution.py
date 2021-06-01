"""
This file is used to evaluate if the stratified sampling is correctly
done by the RandomStratifiedSampler class
"""

from Data.Datasets import PetaleNNDataset
from Data.Sampling import RandomStratifiedSampler
from SQL.constants import *
from SQL.DataManagement.Utils import PetaleDataManager
from Utils.visualization import visualize_class_distribution


DATASET_TYPES = ['train', 'valid', 'test']
LABELS_DICT = {0: "No", 1: "Yes"}


if __name__ == '__main__':

    # STRATIFIED SAMPLING TEST WITH LEARNING 01

    print(f"\nStratified sampling test...\n")

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")
    sampler = RandomStratifiedSampler(1, 1)

    # We save continuous columns and categorical columns
    cont_cols = [AGE_AT_DIAGNOSIS, DT, DOX]
    cat_cols = [SEX, RADIOTHERAPY_DOSE, DEX, BIRTH_AGE, BIRTH_WEIGHT]

    # Creation of dataset
    df = manager.get_table(LEARNING_1)
    datasets = PetaleNNDataset(df, CARDIOMETABOLIC_COMPLICATIONS, cont_cols, cat_cols)

    # Creation of masks
    masks = sampler(datasets.y)

    # Visualization of distribution
    for k, v in masks.items():
        datasets.update_masks(train_mask=v["train"], valid_mask=v["valid"], test_mask=v["test"])
        if v != 'inner':
            for i in DATASET_TYPES:
                visualize_class_distribution(datasets.y[v[i]], LABELS_DICT, title=f"{i}-{k}")
            for inner_k, inner_v in v['inner'].items():
                for j in DATASET_TYPES:
                    visualize_class_distribution(datasets.y[inner_v[j]], LABELS_DICT, title=f"Inner {j}-{inner_k}")

