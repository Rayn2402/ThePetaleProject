"""
Filename: get_obesity_classification_metrics.py

Author: Nicolas Raymond

Description: Adds the sensitivity, specificity and bAcc scores to obesity experiments
             at a given path.
"""
import argparse

from json import load
from os.path import join
from pandas import DataFrame, merge
from src.data.extraction.constants import *
from src.data.extraction.data_management import PetaleDataManager
from src.recording.constants import *
from src.utils.argparsers import print_arguments
from src.utils.score_metrics import Sensitivity, Specificity, BinaryBalancedAccuracy
from src.utils.results_analysis import get_directories
from torch import tensor


def argument_parser():
    """
    This function defines a parser that to extract classification metric scores of each model within an experiment
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 get_obesity_classification_metrics.py [experiment_folder_path]',
                                     description="Adds classification metrics scores to experiment summaries")

    parser.add_argument('-p', '--path', type=str, help='Path of the experiment folder')
    arguments = parser.parse_args()
    print_arguments(arguments)

    return arguments


if __name__ == '__main__':

    args = argument_parser()

    # We initialize the metrics
    metrics = [Sensitivity(), Specificity(), BinaryBalancedAccuracy()]

    # We load the obesity ground truth table
    m = PetaleDataManager()
    gt_df = m.get_table(OBESITY_TARGET)

    # We extract the names of all the folders in the directory
    folders = get_directories(args.path)

    for f in folders:

        # We load the data from the records
        with open(join(args.path, f, RECORDS_FILE), "r") as read_file:
            data = load(read_file)

        # We save the predictions of every participant
        SECTION = 'Section'
        OBESITY_CLASS_PRED = 'OBC'
        pred = {PARTICIPANT: [], 'Section': [], TOTAL_BODY_FAT: [], OBESITY_CLASS_PRED: []}
        for section in [TRAIN_RESULTS, TEST_RESULTS, VALID_RESULTS]:
            for k in data[section].keys():
                pred[PARTICIPANT].append(k)
                pred['Section'].append(section)
                pred[TOTAL_BODY_FAT].append(float(data[section][k][PREDICTION]))
                pred[OBESITY_CLASS_PRED].append(0)

        # We save the predictions in a dataframe
        pred_df = DataFrame(data=pred)

        # We concatenate the dataframes
        pred_df = merge(pred_df, gt_df, on=[PARTICIPANT], how=INNER)

        # We calculate the obesity target predicted
        pred_df.loc[(pred_df[SEX] == 'Women') & (pred_df[AGE] > 18) & (pred_df[TOTAL_BODY_FAT] > 35), [OBESITY_CLASS_PRED]] = 1
        pred_df.loc[(pred_df[SEX] == 'Men') & (pred_df[AGE] > 18) & (pred_df[TOTAL_BODY_FAT] > 25), [OBESITY_CLASS_PRED]] = 1
        pred_df.loc[(pred_df[AGE] < 18) & (pred_df[TOTAL_BODY_FAT] >= OBESITY_PERCENTILE), [OBESITY_CLASS_PRED]] = 1

        # We calculate the metrics
        for s1, s2 in [(TRAIN_RESULTS, TRAIN_METRICS), (TEST_RESULTS, TEST_METRICS), (VALID_RESULTS, VALID_METRICS)]:
            subset_df = pred_df.loc[pred_df[SECTION] == s1, :]
            pred = tensor(subset_df[OBESITY_CLASS_PRED].to_numpy())
            target = tensor(subset_df[OBESITY].astype('float').to_numpy()).long()

            for metric in metrics:
                data[s2][metric.name] = metric(pred=pred, targets=target)


