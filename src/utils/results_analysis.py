"""
Filename: results_analysis.py

Author: Nicolas Raymond

Description: Contains function to help analyses results from different experiments

Date of last modification: 2021/12/01
"""

from json import dump, load
from numpy import mean, std
from os import listdir
from os.path import join, isdir
from pandas import DataFrame
from settings.paths import Paths
from src.recording.constants import MEAN, STD, SUMMARY_FILE, TEST_METRICS
from time import time
from typing import Any, Dict, List, Union


APRIORI_KEYS = ['Support', 'Lift', 'Confidence']


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
        with open(join(path, f, SUMMARY_FILE), "r") as read_file:
            data = load(read_file)

        # We save the metrics in the dictionary
        test_metrics[m] = {metric: f"{round(float(v[MEAN]), nb_digits)} +- {round(float(v[STD]), nb_digits)}"
                           for metric, v in data[TEST_METRICS].items()}

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





