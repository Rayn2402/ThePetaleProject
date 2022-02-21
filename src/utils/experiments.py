"""
Filename: experiments.py

Author: Nicolas Raymond

Description: This file stores experiment functions that can be used with different datasets

Date of last modification: 2022/02/03
"""
from apyori import apriori
from os import mkdir
from os.path import exists, join

from pandas import DataFrame
from settings.paths import Paths
from src.data.processing.datasets import MaskType
from src.data.processing.preprocessing import preprocess_for_apriori
from src.utils.results_analysis import get_apriori_statistics, print_and_save_apriori_rules
from time import time
from typing import Dict, List


def run_apriori_experiment(experiment_name: str,
                           df: DataFrame,
                           target: str,
                           cat_cols: List[str],
                           masks: Dict,
                           arguments,
                           continuous_target: bool = True) -> None:
    """
    Finds association rules among a dataset, using different splits.
    First, all rules are found and saved in a json.
    Then, rules associated to a specified target variable are filtered and saved in a different directory.
    Finally, a summary of rules associated to the target is also saved in the last directory created.

    Args:
        experiment_name: str that will be used to identify directories created during the experiment
        df: pandas dataframe with the records
        target: name of the column associated to the target
        cat_cols: list of categorical columns
        masks: dictionary with idx of train, valid and test set
        arguments: arguments from the apriori argparser
        continuous_target: if True, continuous targets will be converted to classes

    Returns: None
    """
    # We save start time
    start = time()

    # We only keep categorical columns and targets
    df = df[cat_cols + [target]]

    # We save folder names for different results
    result_folder = f"{experiment_name}_apriori"
    filtered_result_folder = f"{result_folder}_{target}"
    f1, f2 = join(Paths.EXPERIMENTS_RECORDS, result_folder), join(Paths.EXPERIMENTS_RECORDS, filtered_result_folder)
    for f in [f1, f2]:
        if not exists(f):
            mkdir(f)

    for i in range(len(masks.keys())):

        # We filter the dataset to only consider training set
        df_subset = df.iloc[masks[i][MaskType.TRAIN]]

        # We preprocess data
        if continuous_target:
            records = preprocess_for_apriori(df_subset, cont_cols={target: arguments.nb_groups}, cat_cols=cat_cols)
        else:
            records = preprocess_for_apriori(df_subset, cat_cols=cat_cols + [target])

        # We print the number of records
        print(f"Number of records : {len(records)}")

        # We run apriori algorithm
        association_rules = apriori(records,
                                    min_support=arguments.min_support,
                                    min_confidence=arguments.min_confidence,
                                    min_lift=arguments.min_lift,
                                    max_length=(arguments.max_length + 1))

        association_results = list(association_rules)

        # We print the number of rules
        print(f"Number of rules : {len(association_results)}")

        # We clean results to only keep association rules of with a single item on the right side
        association_results = [rule for rule in association_results if len(list(rule.ordered_statistics[0].items_add)) < 2]

        # We sort the rules by lift
        association_results = sorted(association_results, key=lambda x: x[2][0][3], reverse=True)

        # We save a dictionary with apriori settings
        settings = {"min_support": arguments.min_support,
                    "min_confidence": arguments.min_confidence,
                    "min_lift": arguments.min_lift,
                    "max_length": arguments.max_length,
                    f"nb_{target}_groups": arguments.nb_groups}

        # We print and save all the rules
        print_and_save_apriori_rules(association_results, settings, f1, f"{result_folder}_{i}", start)

        # We print and save the rules only related to target
        temp_list = []
        for rule in association_results:
            right_part = list(rule.ordered_statistics[0].items_add)
            right_part = right_part[0]
            if right_part.split(" <")[0] == target.upper() or right_part.split(" >")[0] == target.upper():
                temp_list.append(rule)

        association_results = temp_list
        print_and_save_apriori_rules(association_results, settings, f2, f"{filtered_result_folder}_{i}", start, True)

    # We compute summary of apriori results for rules associated to target
    get_apriori_statistics(f2)



