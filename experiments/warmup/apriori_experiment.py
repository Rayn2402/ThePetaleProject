"""
Filename: apriori_experiment.py

Authors: Nicolas Raymond

Description: This file is used to identify meaningful association
             rules between categorical values and VO2 max quantiles.

Date of last modification : 2022/02/02
"""

from apyori import apriori
from os import mkdir
from os.path import dirname, exists, realpath, join
from time import time

import sys


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import MaskType
    from src.data.processing.sampling import extract_masks, GeneChoice, get_warmup_data
    from src.data.processing.preprocessing import preprocess_for_apriori
    from src.utils.argparsers import apriori_argparser
    from src.utils.results_analysis import get_apriori_statistics, print_and_save_apriori_rules

    # Arguments parsing
    args = apriori_argparser()

    # We save start time
    start = time()

    # We first extract data
    manager = PetaleDataManager()
    df, target, cont_cols, cat_cols = get_warmup_data(manager, genes=GeneChoice.ALL, sex=True)

    # Extraction of masks
    masks = extract_masks(Paths.WARMUP_MASK, k=10, l=0)

    # We only keep categorical columns and targets
    df = df[cat_cols + [target]]

    # We save folder names for different results
    f1, f2 = join(Paths.EXPERIMENTS_RECORDS, "warmup_apriori"), join(Paths.EXPERIMENTS_RECORDS, "warmup_apriori_vo2")
    for f in [f1, f2]:
        if not exists(f):
            mkdir(f)

    for i in range(len(masks.keys())):

        df_subset = df.iloc[masks[i][MaskType.TRAIN]]

        # We preprocess data
        records = preprocess_for_apriori(df_subset, cont_cols={target: args.nb_groups}, cat_cols=cat_cols)

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
        print_and_save_apriori_rules(association_results, settings, f1, f'warmup_apriori_{i}', start)

        # We print and save the rules only related to VO2
        temp_list = []
        for rule in association_results:
            right_part = list(rule.ordered_statistics[0].items_add)
            right_part = right_part[0]
            if right_part.split(" <")[0] == target.upper() or right_part.split(" >")[0] == target.upper():
                temp_list.append(rule)

        association_results = temp_list

        print_and_save_apriori_rules(association_results, settings, f2, f'warmup_apriori_vo2_{i}', start, save_genes=True)

    # We compute summary of apriori results for rules associated to V02
    get_apriori_statistics(f2)



