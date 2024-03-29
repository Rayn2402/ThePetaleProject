"""
Filename: results_analysis.py

Author: Nicolas Raymond

Description: Contains function to help analyses results from different experiments

Date of last modification: 2022/04/27
"""

from json import dump, load
from numpy import mean, std
from os import listdir
from os.path import join, isdir
from pandas import DataFrame, merge, read_csv
from settings.paths import Paths
from src.data.extraction.constants import *
from src.data.extraction.data_management import PetaleDataManager
from src.recording.recording import get_evaluation_recap, Recorder
from src.utils.metrics import Sensitivity, Specificity, BinaryBalancedAccuracy
from time import time
from torch import tensor
from typing import Any, Callable, Dict, List, Optional, Union

APRIORI_KEYS = ['Support', 'Lift', 'Confidence']
SECTION = 'Section'
CLASS_PRED = 'CP'
REG_PRED = 'RP'


def extract_predictions(paths: List[str],
                        model_ids: List[str],
                        filename: str) -> None:
    """
    Extracts the predictions of different models and store them in a csv file

    Args:
        paths: list of paths where the records are stored
        model_ids: list of names to identify models from which the predictions are retrieved
        filename: name of the file in which the predictions will be stored

    Returns: None
    """
    predictions = {}
    for i, m, p in zip(range(len(model_ids)), model_ids, paths):

        # We load the data from the records
        with open(join(p, Recorder.RECORDS_FILE), "r") as read_file:
            data = load(read_file)

        if i == 0:
            for k, v in data[Recorder.TEST_RESULTS].items():
                predictions[k] = {Recorder.TARGET: v[Recorder.TARGET], m: v[Recorder.PREDICTION]}
        else:
            for k, v in data[Recorder.TEST_RESULTS].items():
                predictions[k][m] = v[Recorder.PREDICTION]

    # We use the dict to create a dataframe
    df = DataFrame.from_dict(data=predictions, orient='index')
    df.sort_values(Recorder.TARGET, inplace=True)

    # We save the dataframe in a csv
    df.to_csv(path_or_buf=f"{filename}.csv")


def get_classification_metrics(data_manager: Optional[PetaleDataManager],
                               target_table: Optional[str],
                               target_column: str,
                               conditional_columns: Optional[List[str]],
                               experiments_path: str,
                               class_generator_function: Callable
                               ) -> None:
    """
    Calculates the classification metrics related to regression predictions
    according to the criterion in the class generator function

    Args:
        data_manager: data manager to interact with sql database.
                      If no manager is provided, target_column will be computed
        target_table: name of the sql table containing the ground truth
        target_column: name of the column containing targets in the target table
        conditional_columns: columns (other than the predictions) on which the classification depends
        experiments_path: path where the experiment directories are stored
        class_generator_function: function allowing classes to be generated from the
                                  real valued predictions

    Returns: None
    """
    # We initialize the metrics
    metrics = [Sensitivity(), Specificity(), BinaryBalancedAccuracy()]

    # We extract the names of all the folders in the directory
    experiment_folders = get_directories(experiments_path)

    # For each experiment folder, we look at the split folders
    for f1 in experiment_folders:
        sub_path = join(experiments_path, f1)
        for f2 in get_directories(sub_path):

            # We load the data from the records
            with open(join(sub_path, f2, Recorder.RECORDS_FILE), "r") as read_file:
                data = load(read_file)

            # We save the predictions of every participant
            pred = {PARTICIPANT: [], SECTION: [], REG_PRED: [], Recorder.TARGET: []}
            for section in [Recorder.TRAIN_RESULTS, Recorder.TEST_RESULTS, Recorder.VALID_RESULTS]:
                if data.get(section) is not None:
                    for k in data[section].keys():
                        pred[PARTICIPANT].append(k)
                        pred[SECTION].append(section)
                        pred[REG_PRED].append(float(data[section][k][Recorder.PREDICTION]))
                        pred[Recorder.TARGET].append(float(data[section][k][Recorder.TARGET]))

            # We save the predictions in a dataframe
            pred_df = DataFrame(data=pred)

            # We modify the format of conditional columns variables if needed
            conditional_columns = conditional_columns if conditional_columns is not None else []

            if data_manager is not None:

                # We load the obesity ground truth table containing also conditional columns
                df = data_manager.get_table(target_table, [PARTICIPANT, *conditional_columns, target_column])

                # We concatenate the dataframes
                pred_df = merge(pred_df, df, on=[PARTICIPANT], how=INNER)

            else:

                if len(conditional_columns) != 0:

                    # We load the csv with the conditional column needed to generate the ground truth class
                    df = read_csv(join(Paths.DATA, f"{target_table}.csv"), usecols=[PARTICIPANT, *conditional_columns])
                    df[PARTICIPANT] = df[PARTICIPANT].astype(str)

                    # We concatenate the dataframes
                    pred_df = merge(pred_df, df, on=[PARTICIPANT], how=INNER)

                # We calculate the ground truth column
                pred_df = class_generator_function(df=pred_df, input_column=Recorder.TARGET, new_column=target_column)

            # We add the classification prediction
            pred_df = class_generator_function(df=pred_df, input_column=REG_PRED, new_column=CLASS_PRED)

            # We calculate the metrics
            for s1, s2 in [(Recorder.TRAIN_RESULTS, Recorder.TRAIN_METRICS),
                           (Recorder.TEST_RESULTS, Recorder.TEST_METRICS),
                           (Recorder.VALID_RESULTS, Recorder.VALID_METRICS)]:

                if data.get(s2) is not None:
                    subset_df = pred_df.loc[pred_df[SECTION] == s1, :]
                    pred = tensor(subset_df[CLASS_PRED].to_numpy())
                    target = tensor(subset_df[target_column].astype('float').to_numpy()).long()

                    for metric in metrics:
                        data[s2][metric.name] = metric(pred=pred, targets=target)

            # We update the json records file
            with open(join(sub_path, f2, Recorder.RECORDS_FILE), "w") as file:
                dump(data, file, indent=True)

        get_evaluation_recap(evaluation_name='', recordings_path=sub_path)


def get_directories(path: str) -> List[str]:
    """
    Extracts the names of all the folders that can be found at the given path

    Args:
        path: directory

    Returns: list of folder names (str)
    """
    return [folder for folder in listdir(path) if isdir(join(path, folder))]


def get_apriori_statistics(path: str) -> None:
    """
    Calculates the frequencies of rules and mean and std of support, confidence and lift

    Args:
        path: directory containing the json with apriori results

    Returns: None
    """
    # We extract json files with apriori results
    apriori_files = [f for f in listdir(path) if ".json" in f]

    # We retrieve support, confidence and lift of each rules in each file
    rules_statistics = {}
    for f in apriori_files:

        with open(join(path, f), "r") as read_file:
            data = load(read_file)

        for rule in data['Rules'].keys():

            if rules_statistics.get(rule) is None:
                rules_statistics[rule] = {k: [] for k in APRIORI_KEYS}
                rules_statistics[rule]['Count'] = 0

            for k in APRIORI_KEYS:
                rules_statistics[rule][k].append(data['Rules'][rule][k])

            rules_statistics[rule]['Count'] += 1

    # We order rules by count
    rules_statistics = {k: v for k, v in sorted(rules_statistics.items(), key=lambda item: item[1]['Count'],
                                                reverse=True)}

    # We compute the mean and std for each statistics of each rule
    for rule in rules_statistics:
        for k in APRIORI_KEYS:
            rules_statistics[rule][k] = f"{round(mean(rules_statistics[rule][k]).item(), 2)} +-" \
                                        f" {round(std(rules_statistics[rule][k]).item(), 2)}"

    # We save the summary in a json file
    with open(join(path, "summary.json"), "w") as file:
        dump(rules_statistics, file, indent=True)


def get_experiment_summaries(path: str, csv_filename: str, nb_digits: int = 2) -> None:
    """
    From a folder containing experiment results of different models,
    we extract test metrics and save it in a csv that can then be converted to a LaTeX table using :

    https://www.tablesgenerator.com/

    Args:
        path: directory where the experiment results are stored
        csv_filename: name of the csv file containing the extracted data
        nb_digits: number of digits kept in the csv

    Returns: None
    """
    # We extract the names of all the folders in the directory
    folders = get_directories(path)

    # We extract test metrics data from each model directory
    test_metrics = {}
    for f in folders:

        # We extract the name of the model
        m = f.split('_')[0]

        # We load the data from the summary
        with open(join(path, f, Recorder.SUMMARY_FILE), "r") as read_file:
            data = load(read_file)

        # We save the metrics in the dictionary
        test_metrics[m] = {metric: f"{round(float(v[Recorder.MEAN]), nb_digits)} +- {round(float(v[Recorder.STD]), nb_digits)}"
                           for metric, v in data[Recorder.TEST_METRICS].items()}

    # We store the data in a dataframe and save it into a csv
    df = DataFrame(data=test_metrics).T
    df.to_csv(path_or_buf=join(Paths.CSV_FILES, f"{csv_filename}.csv"))


def print_and_save_apriori_rules(rules: List[Any],
                                 settings: Dict[str, Union[float, int]],
                                 folder_path: str,
                                 json_filename: str,
                                 start_time: Any,
                                 save_genes: bool = False) -> None:
    """
    Prints and saves the rules found in a json file

    Args:
        rules: list of rules found with apriori
        settings: dictionary of apriori settings
        folder_path: path of the folder used to store json file with results
        json_filename: name of the json file used to store the results
        start_time: experiment start time
        save_genes: True if we want to save genes involved in rules

    Returns: None
    """
    rules_dictionary = {'Settings': settings, 'Rules': {}}

    for item in rules:

        # We print rule
        rule = f"{list(item.ordered_statistics[0].items_base)} -> {list(item.ordered_statistics[0].items_add)}"
        print(f"Rule : {rule}")

        # We print support
        support = item[1]
        print(f"Support: {support}")

        # We print confidence and lift
        confidence = item[2][0][2]
        lift = item[2][0][3]
        print(f"Confidence: {confidence}")
        print(f"Lift: {lift}")

        # We save statistics in the dictionary
        rules_dictionary['Rules'][rule] = {'Support': support, 'Lift': lift, 'Confidence': confidence}
        print("="*40)

    if save_genes:

        # We initialize a genes counter
        genes_list = []

        for item in rules:
            for chrom_pos_expression in list(item.ordered_statistics[0].items_base):

                # We extract chrom pos and save it
                chrom_pos_splits = chrom_pos_expression.split("_")
                chrom_pos = f"{chrom_pos_splits[0]}_{chrom_pos_splits[1]}"

                if chrom_pos_splits[-1] in ['0/0', '1/1', '0/1']:
                    genes_list.append(chrom_pos)

        rules_dictionary['Genes'] = list(set(genes_list))

    # We save and print the time taken
    time_taken = round((time() - start_time) / 60, 2)
    rules_dictionary['Settings']['time'] = time_taken
    print("Time Taken (minutes): ", time_taken)

    # We save the number of rules
    rules_dictionary['Settings']['nb_of_rules'] = len(rules)

    # We save the dictionary in a json file
    filepath = join(folder_path, f"{json_filename}.json")
    with open(filepath, "w") as file:
        dump(rules_dictionary, file, indent=True)






