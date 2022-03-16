"""
Filename: experiments.py

Author: Nicolas Raymond

Description: This file stores experiment functions that can be used with different datasets

Date of last modification: 2022/02/03
"""
import json

from apyori import apriori
from copy import deepcopy
from os import mkdir
from os.path import exists, join
from pandas import DataFrame
from settings.paths import Paths
from src.data.processing.datasets import MaskType, PetaleDataset
from src.data.processing.preprocessing import preprocess_for_apriori
from src.recording.constants import PREDICTION, RECORDS_FILE, TRAIN_RESULTS, TEST_RESULTS, VALID_RESULTS
from src.recording.recording import Recorder, compare_prediction_recordings, get_evaluation_recap
from src.utils.graph import PetaleGraph, correct_and_smooth
from src.utils.results_analysis import get_apriori_statistics, print_and_save_apriori_rules
from src.utils.score_metrics import RegressionMetric, BinaryClassificationMetric
from time import time
from torch import zeros
from tqdm import tqdm
from typing import Dict, List, Optional, Union


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


def run_correct_and_smooth_experiment(dataset: PetaleDataset,
                                      evaluation_name: str,
                                      masks: dict,
                                      metrics: Union[List[BinaryClassificationMetric], List[RegressionMetric]],
                                      path: str,
                                      r_smooth: float,
                                      r_correct: float,
                                      max_degree: Optional[int] = None,
                                      include_distances: bool = False,
                                      nb_iter: int = 1):

    # For all splits
    for k, m in tqdm(masks.items()):

        # We extract the records from folder "Split k"
        with open(join(path, f"Split_{k}", RECORDS_FILE)) as json_file:
            records_k = json.load(json_file)

        # We save the predictions made for each id
        pred = zeros((len(dataset), 1))
        for result_section in [TRAIN_RESULTS, TEST_RESULTS, VALID_RESULTS]:
            for id_, result_dict in records_k[result_section].items():
                pred[dataset.ids_to_row_idx[id_]] = float(result_dict[PREDICTION])

        # We update dataset mask
        dataset.update_masks(train_mask=m[MaskType.TRAIN],
                             test_mask=m[MaskType.TEST],
                             valid_mask=m[MaskType.VALID])

        # Recorder initialization
        recorder = Recorder(evaluation_name=evaluation_name,
                            index=k, recordings_path=Paths.EXPERIMENTS_RECORDS)

        # We build the graph
        g = PetaleGraph(dataset, include_distances=include_distances,
                        cat_cols=dataset.cat_cols, max_degree=max_degree)

        # We proceed to correction and smoothing of the predictions
        y_copy = deepcopy(dataset.y)
        cs_pred = correct_and_smooth(g, pred=pred, labels=y_copy, masks=m, r_correct=r_correct,
                                     r_smooth=r_smooth, nb_iter=nb_iter)

        for mask, masktype in [(m[MaskType.TRAIN], MaskType.TRAIN),
                               (m[MaskType.TEST], MaskType.TEST),
                               (m[MaskType.VALID], MaskType.VALID)]:
            if mask is not None:

                # We record predictions
                pred, ground_truth = cs_pred[mask], dataset.y[mask]
                recorder.record_predictions([dataset.ids[i] for i in mask], pred, ground_truth, mask_type=masktype)

                # We record scores
                for metric in metrics:
                    recorder.record_scores(score=metric(pred, ground_truth), metric=metric.name, mask_type=masktype)

        # Generation of the file with the results
        recorder.generate_file()
        compare_prediction_recordings(evaluations=[evaluation_name], split_index=k,
                                      recording_path=Paths.EXPERIMENTS_RECORDS)

    get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)



