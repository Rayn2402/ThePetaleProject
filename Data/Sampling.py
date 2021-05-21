"""
Author : Nicolas Raymond

This file contains the Sampler class used to separate test sets from train sets
"""

from numpy import array
from numpy.random import seed
from pandas import qcut
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
    def __init__(self, n_out_split: int, n_in_split: int,
                 valid_size: float = 0.20, test_size: float = 0.20, random_state: Optional[int] = None):
        """
        Set public attributes of the sampler

        Args:
            n_out_split: number of outer splits to produce
            n_in_split: number of inner splits to produce
            valid_size: percentage of data taken to create the validation indexes set
            test_size: percentage of data taken to create the train indexes set
        """
        assert n_out_split > 0, 'Number of outer split must be greater than 0'
        assert n_in_split >= 0, 'Number of inner split must be greater or equal to 0'
        assert 0 <= valid_size < 1, 'Validation size must be in the range [0, 1)'
        assert 0 < test_size < 1, 'Test size must be in the range (0, 1)'
        assert valid_size + test_size < 1, 'Train size must be non null'

        # Public attributes
        self.n_out_split = n_out_split
        self.n_in_split = n_in_split
        self.random_state = random_state

        # Public method
        self.split = self.__define_split_function(test_size, valid_size)

    def __call__(self, targets: Union[tensor, array, List[Any]]
                 ) -> Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]:
        """
        Returns lists of indexes to use as train, valid and test masks for outer and inner validation loops.
        The proportion of each class is conserved within each split.

        Args:
            targets: sequence of float/int used for stratification

        Returns: Dictionary of dictionaries with list of indexes.

        Example:

        {0: {'train': [..], 'valid': [..], 'test': [..], 'inner': {0: {'train': [..], 'valid': [..], 'test': [..] }}}

        """

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

    @staticmethod
    def __define_split_function(test_size, valid_size) -> Callable:
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
                remaining_idx, test_mask = train_test_split(idx, stratify=targets[idx], test_size=test_size)
                train_mask, valid_mask = train_test_split(remaining_idx, stratify=targets[remaining_idx],
                                                          test_size=valid_size)

                return {"train": train_mask, "valid": valid_mask, "test": test_mask}
        else:
            # Split must extract train and test masks only
            def split(idx: array, targets: array) -> Dict[str, array]:
                train_mask, test_mask = train_test_split(idx, stratify=targets[idx], test_size=test_size)

                return {"train": train_mask, "valid": None, "test": test_mask}

        return split

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

