"""
Filename: apriori_experiment.py

Authors: Nicolas Raymond

Description: This file is used to identify meaningful association
             rules between categorical values and VO2 max quantiles.

Date of last modification : 2021/11/05
"""

from apyori import apriori
from os.path import dirname, realpath, join
from time import time
from typing import Dict, List, Union, Any

import argparse
import json
import sys


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python apriori_experiment.py',
                                     description="Runs the apriori algorithm on the warmup dataset")

    # Nb inner split and nb outer split selection
    parser.add_argument('-min_sup', '--min_support', type=float, default=0.1,
                        help='Minimal support value (default = 0.1)')
    parser.add_argument('-min_conf', '--min_confidence', type=float, default=0.60,
                        help='Minimal confidence value (default = 0.60)')
    parser.add_argument('-min_lift', '--min_lift', type=float, default=1.20,
                        help='Minimal lift value (default = 1.20)')
    parser.add_argument('-max_length', '--max_length', type=int, default=1,
                        help='Max cardinality of item sets at the left side of rules (default = 1)')
    parser.add_argument('-nb_groups', '--nb_groups', type=int, default=2,
                        help='Number quantiles considered to create V02 groups (default = 2)')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


def print_and_save_rules(rules: List[Any],
                         settings: Dict[str, Union[float, int]],
                         json_filename: str,
                         start_time: Any,
                         save_genes: bool = False) -> None:
    """
    Prints and saves the rules found in a json file

    Args:
        rules: list of rules found with apriori
        settings: dictionary of apriori settings
        json_filename: name of the json file used to store the results
        start_time: experiment start time
        save_genes: True if we want to save genes involved in rules

    Returns: None
    """
    rules_dictionary = {'Settings': settings}

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
        rules_dictionary[rule] = {'Support': support, 'Lift': lift, 'Confidence': confidence}
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
    filepath = join(Paths.EXPERIMENTS_RECORDS, f"{json_filename}.json")
    with open(filepath, "w") as file:
        json.dump(rules_dictionary, file, indent=True)


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.sampling import GeneChoice, get_warmup_data
    from src.data.processing.preprocessing import preprocess_for_apriori

    # Arguments parsing
    args = argument_parser()

    # We save start time
    start = time()

    # We first extract data
    manager = PetaleDataManager()
    df, target, cont_cols, cat_cols = get_warmup_data(manager, genes=GeneChoice.ALL, sex=True)

    # We only keep categorical columns and targets
    df = df[cat_cols + [target]]

    # We preprocess data
    records = preprocess_for_apriori(df, cont_cols={target: args.nb_groups}, cat_cols=cat_cols)

    # We print the number of itemsets
    print(f"Number of records : {len(records)}")

    # We run apriori algorithm
    association_rules = apriori(records, min_support=args.min_support, min_confidence=args.min_confidence,
                                min_lift=args.min_lift, max_length=(args.max_length + 1))
    association_results = list(association_rules)

    # We print the number of rules
    print(f"Number of rules : {len(association_results)}")

    # We clean results to only keep association rules of type (... -> VO2)
    association_results = [rule for rule in association_results if len(list(rule.ordered_statistics[0].items_add)) < 2]

    # We sort the rules by lift
    association_results = sorted(association_results, key=lambda x: x[2][0][3], reverse=True)

    # We save a dictionary with apriori settings
    settings = {'min_support': args.min_support, 'min_confidence': args.min_confidence,
                'min_lift': args.min_lift, 'max_length': args.max_length, 'nb_VO2_groups': args.nb_groups}

    # We print and save all the rules
    print_and_save_rules(association_results, settings, 'warmup_apriori', start)

    # We print and save the rules only related to VO2
    temp_list = []
    for rule in association_results:
        right_part = list(rule.ordered_statistics[0].items_add)
        right_part = right_part[0]
        if right_part.split(" <")[0] == target.upper():
            temp_list.append(rule)

    association_results = temp_list

    print_and_save_rules(association_results, settings, 'warmup_apriori_vo2', start, save_genes=True)



