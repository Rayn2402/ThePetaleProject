"""
Filename: results_analysis.py

Author: Nicolas Raymond

Description: Contains function to help analyses results from different experiments

Date of last modification: 2021/12/01
"""

from json import dump, load
from numpy import mean, std
from os import listdir
from os.path import join

APRIORI_KEYS = ['Support', 'Lift', 'Confidence']


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


