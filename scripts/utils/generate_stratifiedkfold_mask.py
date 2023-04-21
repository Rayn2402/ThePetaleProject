"""
Filename: generate_stratified_rss_masks.py

Author: Nicolas Raymond

Description: Script used to produce train, valid and test masks related
             to stratified k-fold splits on the VO2 dataset.

Date of last modification: --
"""
import sys
from json import dump
from os.path import dirname, join, realpath
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from settings.paths import Paths
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.datasets import PetaleDataset, MaskType
from src.data.processing.sampling import get_VO2_data, RandomStratifiedSampler

if __name__ == '__main__':

    # We retrieve the data needed
    df, target, cont_cols, cat_cols = get_VO2_data(PetaleDataManager())

    # We create a dataset
    dts = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

    # We extract the preprocess data
    x, _, idx = dts[:]

    # We save the idx of the column associated to sex category
    j = dts.cat_idx[0]

    # We initialize a dictionary to store the split indexes
    masks = {i: {} for i in range(5)}

    for i, (remaining_idx, test_idx) in enumerate(StratifiedKFold(n_splits=5).split(x, x[:, j])):

        # We extract the ids of the test set (stratified by sex)
        masks[i][MaskType.TEST] = test_idx.tolist()

        # We divide the remaining idx into training and valid set (stratified by sex)
        train_idx, valid_idx = train_test_split(remaining_idx, train_size=0.80)
        masks[i][MaskType.TRAIN] = train_idx.tolist()
        masks[i][MaskType.VALID] = valid_idx.tolist()

        # We repeat the process using only the data in the training set and the valid set
        inner_skf = StratifiedKFold(n_splits=5)
        inner_idx = train_idx.tolist() + valid_idx.tolist()
        masks[i][MaskType.INNER] = {}
        for k, (inner_remaining_idx, inner_test_idx) in enumerate(inner_skf.split(x[inner_idx, :], x[inner_idx, j])):

            # We extract the inner test idx
            masks[i][MaskType.INNER][k] = {MaskType.TEST: inner_test_idx.tolist()}

            # We extract the inner train and valid idx
            inner_train_idx, inner_valid_idx = train_test_split(inner_remaining_idx, train_size=0.80)
            masks[i][MaskType.INNER][k][MaskType.TRAIN] = inner_train_idx.tolist()
            masks[i][MaskType.INNER][k][MaskType.VALID] = inner_valid_idx.tolist()

    # We look at the mask
    RandomStratifiedSampler.visualize_splits(masks)

    # We save the dictionary in a json file
    with open(join(Paths.MASKS, 'vo2_mask.json'), 'w') as file:
        dump(masks, file, indent=True)
