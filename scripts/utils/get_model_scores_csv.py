"""
Filename: get_model_scores_csv.py

Author: Nicolas Raymond

Description: Creates a csv with scores of test metrics of each model compared within an experiment

Date of last modification: 2021/12/2
"""
import argparse

from src.utils.results_analysis import get_experiment_summaries


def argument_parser():
    """
    This function defines a parser that to extract scores of each model within an experiment
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 get_model_scores_csv.py [experiment_folder_path]',
                                     description="Creates a csv with metrics scores of each model")

    parser.add_argument('-p', '--path', type=str, help='Path of the experiment folder')

    parser.add_argument('-fn', '--filename', type=str, help='Name of the csv file')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


if __name__ == '__main__':
    args = argument_parser()
    get_experiment_summaries(path=args.path, csv_filename=args.filename)
