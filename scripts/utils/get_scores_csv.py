"""
Filename: get_scores_csv.py

Author: Nicolas Raymond

Description: Script that creates a csv with scores of test metrics of each model compared within an experiment

Date of last modification: 2022/07/13
"""
import sys
from argparse import ArgumentParser
from os.path import dirname, realpath

# Imports specific to project
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from src.utils.results_analyses import get_experiment_summaries


def argument_parser():
    """
    This function defines a parser that to extract scores of each model within an experiment
    """
    # Create a parser
    parser = ArgumentParser(usage='\n python get_scores_csv.py [experiment_folder_path]',
                            description="Creates a csv with metrics scores of each model")

    parser.add_argument('-p', '--path', type=str, help='Path of the experiment folder')

    parser.add_argument('-fn', '--filename', type=str, help='Name of the csv file')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    args = argument_parser()
    get_experiment_summaries(path=args.path, csv_filename=args.filename)
