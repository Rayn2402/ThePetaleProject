"""
Author : Nicolas Raymond

This file contains the Sampler class used to separate test sets from train sets
"""

from src.data.processing.datasets import PetaleDataset
from itertools import product
from json import load
from numpy import array
from numpy.random import seed
from pandas import qcut, DataFrame
from sklearn.model_selection import train_test_split
from src.data.extraction.constants import *
from src.data.extraction.data_management import PetaleDataManager
from torch import tensor
from tqdm import tqdm
from typing import List, Union, Optional, Dict, Any, Tuple, Callable

TRAIN, VALID, TEST, INNER = "train", "valid", "test", "inner"
MASK_TYPES = [TRAIN, VALID, TEST]
SIGNIFICANT, ALL = "significant", "all"
GENES_CHOICES = [SIGNIFICANT, ALL]
COMPLICATIONS_CHOICES = [CARDIOMETABOLIC_COMPLICATIONS, BONE_COMPLICATIONS, NEUROCOGNITIVE_COMPLICATIONS, COMPLICATIONS]


class RandomStratifiedSampler:
    """
    Object uses in order to generate lists of indexes to use as train, valid
    and test masks for outer and inner validation loops.
    """
    def __init__(self, dataset: PetaleDataset,
                 n_out_split: int, n_in_split: int, valid_size: float = 0.20, test_size: float = 0.20,
                 random_state: Optional[int] = None, alpha: int = 4, patience: int = 100):
        """
        Set private and public attributes of the sampler

        Args:
            n_out_split: number of outer splits to produce
            n_in_split: number of inner splits to produce
            valid_size: percentage of data taken to create the validation indexes set
            test_size: percentage of data taken to create the train indexes set
            alpha: IQR multiplier used to check numerical variable range validity of the masks created
            patience: number of tries that the sampler has to make a single valid split
        """
        assert n_out_split > 0, 'Number of outer split must be greater than 0'
        assert n_in_split >= 0, 'Number of inner split must be greater or equal to 0'
        assert 0 <= valid_size < 1, 'Validation size must be in the range [0, 1)'
        assert 0 < test_size < 1, 'Test size must be in the range (0, 1)'
        assert valid_size + test_size < 1, 'Train size must be non null'

        # Private attributes
        self.__dataset = dataset
        if self.__dataset.encodings is not None:
            self.__unique_encodings = {k: list(v.values()) for k, v in self.__dataset.encodings.items()}
        else:
            self.__unique_encodings = {}

        # Public attributes
        self.alpha = alpha
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
        targets = targets if self.is_categorical(targets) else self.mimic_classes(targets)

        # We set the random state
        if self.random_state is not None:
            seed(self.random_state)

        # We initialize the dict that will contain the results and the list of indexes to use
        masks, idx = {}, array(range(len(targets)))

        # We save a copy of targets in an array
        targets_c = array(targets)

        with tqdm(total=(self.n_out_split + self.n_out_split*self.n_in_split)) as bar:
            for i in range(self.n_out_split):

                # We create outer split masks
                masks[i] = {**self.split(idx, targets_c), INNER: {}}
                bar.update()

                for j in range(self.n_in_split):

                    # We create the inner split masks
                    masks[i][INNER][j] = self.split(masks[i][TRAIN], targets_c)
                    bar.update()

        # We turn arrays of idx into lists of idx
        self.serialize_masks(masks)

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
                return {TRAIN: train_mask, VALID: valid_mask, TEST: test_mask}
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
                return {TRAIN: train_mask, VALID: None, TEST: test_mask}

        return split

    def check_masks_validity(self, train_mask: List[int], test_mask: List[int],
                             valid_mask: Optional[List[int]] = None) -> bool:
        """
        Valid if categorical and numerical variables of other masks are out of the range of train mask

        Args:
            train_mask: List of idx to use for training
            test_mask: list of idx to use for test
            valid_mask: list of idx to use for validation

        Returns: True if the masks are valid
        """
        # We update the masks of the dataset
        self.__dataset.update_masks(train_mask, test_mask, valid_mask)

        # We extract train dataframe
        imputed_df = self.__dataset.get_imputed_dataframe()
        train_df = imputed_df.iloc[train_mask]

        # We check if all categories of categorical columns are in the training set
        for cat, values in self.__unique_encodings.items():
            for c in train_df[cat].unique():
                if c not in values:
                    return False

        # We save q1 and q3 of each numerical columns
        train_quantiles = {c: (train_df[c].quantile(0.25), train_df[c].quantile(0.75))
                           for c in self.__dataset.cont_cols}

        # We validate the other masks
        other_masks = [m for m in [valid_mask, test_mask] if m is not None]
        for mask in other_masks:

            # We extract the subset
            subset_df = imputed_df.iloc[mask]

            # We check if all numerical values are not extreme outliers according to the train mask
            for cont_col, (q1, q3) in train_quantiles.items():
                iqr = q3 - q1
                other_min, other_max = (subset_df[cont_col].min(), subset_df[cont_col].max())
                if other_min < q1 - self.alpha*iqr or other_max > q3 + self.alpha*iqr:
                    return False

            return True

    @staticmethod
    def is_categorical(targets: Union[tensor, array]) -> bool:
        """
        Check if the number of unique values is lower than the quarter of the length of the targets sequence

        Args:
            targets: sequence of float/int used for stratification

        Returns: bool
        """
        target_list_copy = list(targets)
        return len(set(target_list_copy)) < 0.15*len(target_list_copy)

    @staticmethod
    def mimic_classes(targets: Union[tensor, array, List[Any]]) -> array:
        """
        Creates fake classes array out of real-valued targets sequence using quartiles
        Args:
            targets: sequence of float/int used for stratification

        Returns: array with fake classes
        """
        return qcut(array(targets), 2, labels=False)

    @staticmethod
    def serialize_masks(masks: Dict[int, Dict[str, Union[array, Dict[str, array]]]]) -> None:
        """
        Turns all numpy arrays of idx into lists of idx

        Args:
            masks: dictionary of masks

        Returns: None

        """
        for k, v in masks.items():
            for t1 in MASK_TYPES:
                masks[k][t1] = v[t1].tolist() if v[t1] is not None else None
            for in_k, in_v in masks[k][INNER].items():
                for t2 in MASK_TYPES:
                    masks[k][INNER][in_k][t2] = in_v[t2].tolist() if in_v[t2] is not None else None

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
            print(INNER)
            for k1, v1 in v['inner'].items():
                valid = v1['valid'] if v1['valid'] is not None else []
                print(f"{k+1}.{k1} -> Train {len(v1['train'])} - Valid {len(valid)} -"
                      f" Test {len(v1['test'])}")
            print("#----------------------------------#")


def extract_masks(file_path: str, k: int = 20, l: int = 20
                  ) -> Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]:
    """
    Extract masks saved in json file

    Args:
        file_path: path of json file containing the masks
        k: number of outer loops to extract
        l: number of inner loops to extract

    Returns: masks
    """
    # Opening JSON file
    f = open(file_path)

    # Extract complete masks
    all_masks = load(f)

    # Extraction of masks subset
    masks = {}
    for i in map(str, range(k)):
        int_i = int(i)
        masks[int_i] = {}
        for t in MASK_TYPES:
            masks[int_i][t] = all_masks[i][t]
        masks[int_i][INNER] = {}
        for j in map(str, range(l)):
            int_j = int(j)
            masks[int_i][INNER][int_j] = {}
            for t in MASK_TYPES:
                masks[int_i][INNER][int_j][t] = all_masks[i][INNER][j][t]

    # Closing file
    f.close()

    return masks


def push_valid_to_train(masks: Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]
                        ) -> Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]:
    """
    Pushes all index of validation masks into train masks

    Args:
        masks: dictionary with list of idx to use as train, valid and test masks

    Returns: same masks with valid idx added to test idx
    """
    for k, v in masks.items():
        masks[k][TRAIN] += v[VALID]
        masks[k][VALID] = None
        for in_k, in_v in masks[k][INNER].items():
            masks[k][INNER][in_k][TRAIN] += in_v[VALID]
            masks[k][INNER][in_k][VALID] = None


def get_learning_one_data(data_manager: PetaleDataManager, baselines: bool,
                          complications: List[str], genes: Optional[str] = None
                          ) -> Tuple[DataFrame, Optional[List[str]], Optional[List[str]]]:
    """
    Extract dataframe needed to proceed to learning one experiments and turn it into a dataset

    Args:
        data_manager: data manager to communicate with the database
        baselines: True if we want to include baselines variables
        complications: one complication choice among :
         - CARDIOMETABOLIC_COMPLICATIONS
         - BONE_COMPLICATIONS
         - NEUROCOGNITIVE_COMPLICATIONS
         - COMPLICATIONS
        genes: one choice among (None, "significant", "all")

    Returns: dataframe, continuous columns, categorical columns
    """
    for c in complications:
        assert c in COMPLICATIONS_CHOICES, f"complications values must be in {COMPLICATIONS_CHOICES}"

    # We initialize empty lists for continuous and categorical columns
    cont_cols, cat_cols = [], []

    # We check for baselines
    if baselines:
        cont_cols += [AGE_AT_DIAGNOSIS, DT, DOX]
        cat_cols += [SEX, RADIOTHERAPY_DOSE, DEX, BIRTH_AGE, BIRTH_WEIGHT]

    # We check for genes
    if genes is not None:
        assert genes in GENES_CHOICES, f"genes value must be in {GENES_CHOICES}"
        if genes == SIGNIFICANT:
            cat_cols += SIGNIFICANT_CHROM_POS
        else:
            cat_cols += ALL_CHROM_POS

    # We extract the dataframe
    df = data_manager.get_table(LEARNING_1, columns=[PARTICIPANT] + complications + cont_cols + cat_cols)

    # We change format of columns list if they are empty
    cont_cols = cont_cols if len(cont_cols) > 0 else None
    cat_cols = cat_cols if len(cat_cols) > 0 else None

    return df, cont_cols, cat_cols


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

    return multitask_labels, labels_dict


def get_warmup_data(data_manager: PetaleDataManager, genes: Optional[str] = None
                    ) -> Tuple[DataFrame, str, Optional[List[str]], Optional[List[str]]]:
    """
    Extract dataframe needed to proceed to warmup experiments and turn it into a dataset

    Args:
        data_manager: data manager to communicate with the database
        genes: one choice among (None, "significant", "all")

    Returns: dataframe, target, continuous columns, categorical columns
    """
    # We save the name of continuous columns in a list
    cont_cols = [WEIGHT, TDM6_HR_END, TDM6_DIST, DT, AGE, MVLPA]

    # We check for genes
    if genes is not None:
        assert genes in GENES_CHOICES, f"genes value must be in {GENES_CHOICES}"
        if genes == SIGNIFICANT:
            genes = SIGNIFICANT_CHROM_POS
        else:
            genes = ALL_CHROM_POS_WARMUP

    # We extract the dataframe
    df = data_manager.get_table(LEARNING_0_GENES, columns=[PARTICIPANT, VO2R_MAX] + cont_cols + genes)

    return df, VO2R_MAX, cont_cols, genes

