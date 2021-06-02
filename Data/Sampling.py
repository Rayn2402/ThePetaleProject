"""
Author : Nicolas Raymond

This file contains the Sampler class used to separate test sets from train sets
"""

from Data.Datasets import PetaleRFDataset
from itertools import product
from numpy import array
from numpy.random import seed
from pandas import qcut, DataFrame
from sklearn.model_selection import train_test_split
from torch import tensor
from typing import List, Union, Optional, Dict, Any, Tuple, Callable

SIGNIFICANT, ALL = "significant", "all"
GENES_CHOICES = [None, SIGNIFICANT, ALL]


class RandomStratifiedSampler:
    """
    Object uses in order to generate lists of indexes to use as train, valid
    and test masks for outer and inner validation loops.
    """
    def __init__(self, dataset: PetaleRFDataset,
                 n_out_split: int, n_in_split: int, valid_size: float = 0.20, test_size: float = 0.20,
                 random_state: Optional[int] = None, patience: int = 100):
        """
        Set private and public attributes of the sampler

        Args:
            n_out_split: number of outer splits to produce
            n_in_split: number of inner splits to produce
            valid_size: percentage of data taken to create the validation indexes set
            test_size: percentage of data taken to create the train indexes set
            patience: number of tries that the sampler has to make a single valid split
        """
        assert n_out_split > 0, 'Number of outer split must be greater than 0'
        assert n_in_split >= 0, 'Number of inner split must be greater or equal to 0'
        assert 0 <= valid_size < 1, 'Validation size must be in the range [0, 1)'
        assert 0 < test_size < 1, 'Test size must be in the range (0, 1)'
        assert valid_size + test_size < 1, 'Train size must be non null'

        # Private attributes
        self.__dataset = dataset

        # Public attributes
        self.n_out_split = n_out_split
        self.n_in_split = n_in_split
        self.patience = patience
        self.random_state = random_state

        # Public method
        self.split = self.__define_split_function(test_size, valid_size)

    def __call__(self, stratify: Optional[Union[array, tensor]] = None,
                 ) -> Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]:
        """
        Returns lists of indexes to use as train, valid and test masks for outer and inner validation loops.
        The proportion of each class is conserved within each split.

        Args:
            stratify: array or tensor used for stratified split (if None, dataset.y will be used)


        Returns: Dictionary of dictionaries with list of indexes.

        Example:

        {0: {'train': [..], 'valid': [..], 'test': [..], 'inner': {0: {'train': [..], 'valid': [..], 'test': [..] }}}

        """
        # We set targets to use for stratification
        targets = self.__dataset.y if stratify is None else stratify

        # We set the random state
        if self.random_state is not None:
            seed(self.random_state)

        # We initialize the dict that will contain the results and the list of indexes to use
        masks, idx = {}, array(range(len(targets)))

        # We save a copy of targets in an array
        targets_c = array(targets)

        for i in range(self.n_out_split):

            # We create outer split masks
            masks[i] = {**self.split(idx, targets_c), "inner": {}}

            for j in range(self.n_in_split):

                # We create the inner split masks
                masks[i]["inner"][j] = self.split(masks[i]["train"], targets_c)

        return masks

    def __define_split_function(self, test_size, valid_size) -> Callable:
        """
        Defines the split function according to the valid size
        Args:
            valid_size: percentage of data taken to create the validation indexes set
            test_size: percentage of data taken to create the train indexes set

        Returns: split function
        """
        if valid_size > 0:

            # Split must extract train, valid and test masks
            def split(idx: array, targets: array) -> Dict[str, array]:

                # We initialize loop important values
                mask_ok = False
                nb_tries_remaining = self.patience

                # We test multiple possibilities till we find one or the patience is achieved
                while not mask_ok and nb_tries_remaining > 0:
                    remaining_idx, test_mask = train_test_split(idx, stratify=targets[idx], test_size=test_size)
                    train_mask, valid_mask = train_test_split(remaining_idx, stratify=targets[remaining_idx],
                                                              test_size=valid_size)
                    mask_ok = self.check_masks_validity(train_mask, test_mask, valid_mask)
                    nb_tries_remaining -= 1

                assert mask_ok, "The sampler could not find a proper train, valid and test split"
                return {"train": train_mask, "valid": valid_mask, "test": test_mask}
        else:
            # Split must extract train and test masks only
            def split(idx: array, targets: array) -> Dict[str, array]:

                # We initialize loop important values
                mask_ok = False
                nb_tries_remaining = self.patience

                # We test multiple possibilities till we find one or the patience is achieved
                while not mask_ok and nb_tries_remaining > 0:
                    train_mask, test_mask = train_test_split(idx, stratify=targets[idx], test_size=test_size)
                    mask_ok = self.check_masks_validity(train_mask, test_mask)
                    nb_tries_remaining -= 1

                assert mask_ok, "The sampler could not find a proper train, valid split"
                return {"train": train_mask, "valid": None, "test": test_mask}

        return split

    def check_masks_validity(self, train_mask: List[int], test_mask: List[int],
                             valid_mask: Optional[List[int]] = None) -> bool:
        """
        Valid if categorical and numerical variables of other masks are out of the range of train mask

        Args:
            train_mask: idx to use for training
            test_mask: list of idx to use for test
            valid_mask: list of idx to use for validation

        Returns: True if the masks are valid
        """
        # We update the masks of the dataset
        self.__dataset.update_masks(train_mask, test_mask, valid_mask)

        # We extract train dataframe
        train_df = self.__dataset.x.iloc[train_mask]

        # We save unique values of categorical columns
        unique_train_cats = {c: list(train_df[c].unique()) for c in self.__dataset.cat_cols}

        # # We save min and max of each numerical columns
        train_quantiles = {c: (train_df[c].quantile(0.25), train_df[c].quantile(0.75))
                           for c in self.__dataset.cont_cols}

        # We validate the other masks
        other_masks = [m for m in [valid_mask, test_mask] if m is not None]
        for mask in other_masks:

            # We extract the subset
            subset_df = self.__dataset.x.iloc[mask]

            # # We check if all numerical values are not extreme outliers according to the train mask
            for cont_col, (q1, q3) in train_quantiles.items():
                iqr = q3 - q1
                other_min, other_max = (subset_df[cont_col].min(), subset_df[cont_col].max())
                if other_min < q1 - 3*iqr or other_max > q3 + 3*iqr:
                    # print("Numerical range not satisfied")
                    return False

            # We check if all categories seen in the other mask is present in the train mask
            for cat_col, values in unique_train_cats.items():
                unique_other_cats = list(subset_df[cat_col].unique())
                for c in unique_other_cats:
                    if c not in values:
                        # print(f"Category {c} of variable {cat_col} not in the train set")
                        return False

            return True

    @staticmethod
    def is_categorical(targets: Union[tensor, array, List[Any]]) -> bool:
        """
        Check if the number of unique values is greater than the quarter of the length of the targets sequence

        Args:
            targets: sequence of float/int used for stratification

        Returns: bool
        """
        target_list_copy = list(targets)
        return len(set(target_list_copy)) > 0.25*len(target_list_copy)

    @staticmethod
    def mimic_classes(targets: Union[tensor, array, List[Any]]) -> array:
        """
        Creates fake classes array out of real-valued targets sequence using quartiles
        Args:
            targets: sequence of float/int used for stratification

        Returns: array with fake classes
        """
        return qcut(array(targets), 4, labels=False)

    @staticmethod
    def visualize_splits(datasets: dict) -> None:
        """
        Details the data splits for the experiment

        :param datasets: dict with all datasets obtain from the Sampler
        """
        print("#----------------------------------#")
        for k, v in datasets.items():
            print(f"Split {k+1} \n")
            print(f"Outer :")
            valid = v['valid'] if v['valid'] is not None else []
            print(f"Train {len(v['train'])} - Valid {len(valid)} - Test {len(v['test'])}")
            print("Inner")
            for k1, v1 in v['inner'].items():
                valid = v1['valid'] if v1['valid'] is not None else []
                print(f"{k+1}.{k1} -> Train {len(v1['train'])} - Valid {len(valid)} -"
                      f" Test {len(v1['test'])}")
            print("#----------------------------------#")


def generate_multitask_labels(df: DataFrame, target_columns: List[str]) -> Tuple[array, Dict[int, tuple]]:
    """
    Generates single array of class labels using all possible combinations of unique values
    contained within target_columns.

    For example, for 3 binary columns we will generate 2^3 = 8 different class labels and assign them
    to the respective rows.

    Args:
        df: dataframe with items to classify
        target_columns: names of the columns to use for multitask learning

    Returns: array with labels, dict with the meaning of each label
    """
    # We extract unique values of each target column
    possible_targets = [list(df[target].unique()) for target in target_columns]

    # We generate all possible combinations of these unique values and assign them a label
    labels_dict = {combination: i for i, combination in enumerate(product(*possible_targets))}

    # We associate labels to the items in the dataframe
    item_labels_union = list(zip(*[df[t].values for t in target_columns]))
    multitask_labels = array([labels_dict[item] for item in item_labels_union])

    # We rearrange labels_dict for visualization purpose
    labels_dict = {v: k for k, v in labels_dict.items()}

    return tensor(multitask_labels), labels_dict


# def get_warmup_sampler(dm: PetaleDataManager, to_dataset: bool = True):
#     """
#     Creates a Sampler for the WarmUp data table
#     :param dm: PetaleDataManager
#     :param to_dataset: bool indicating if we want a PetaleDataset (True) or a PetaleDataframe (False)
#     """
#     cont_cols = [WEIGHT, TDM6_HR_END, TDM6_DIST, DT, AGE, MVLPA]
#     return Sampler(dm, LEARNING_0, cont_cols, VO2R_MAX, to_dataset=to_dataset)
#
#
# def get_learning_one_sampler(dm: PetaleDataManager, to_dataset: bool = True, genes: Optional[str] = None):
#     """
#     Creates a Sampler for the L1 data table
#
#     :param dm: PetaleDataManager
#     :param to_dataset: bool indicating if we want a PetaleDataset (True) or a PetaleDataframe (False)
#     :param genes: str indicating if we want to consider no genes, significant genes or all genes
#     """
#     assert genes in GENES_CHOICES, f"Genes parameter must be in {GENES_CHOICES}"
#
#     # We save table name
#     table_name = LEARNING_1
#
#     # We save continuous columns
#     cont_cols = [AGE_AT_DIAGNOSIS, DT, DOX]
#
#     # We save the categorical columns
#     cat_cols = [SEX, RADIOTHERAPY_DOSE, DEX, BIRTH_AGE, BIRTH_WEIGHT]
#
#     if genes == SIGNIFICANT:
#         table_name = LEARNING_1_1
#         cat_cols = cat_cols + SIGNIFICANT_CHROM_POS
#
#     elif genes == ALL:
#         table_name = LEARNING_1_2
#         current_col_list = [PARTICIPANT, CARDIOMETABOLIC_COMPLICATIONS] + cont_cols + cat_cols
#         cat_cols = cat_cols + [c for c in dm.get_column_names(table_name) if
#                                c not in current_col_list]
#
#     return Sampler(dm, table_name, cont_cols, CARDIOMETABOLIC_COMPLICATIONS, cat_cols, to_dataset)

