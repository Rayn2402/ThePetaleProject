"""
This file is used as a script to produce train, valid and test masks related to a learning table
"""

import argparse
import os
import sys

# Few lines of code to avoid problem when calling script from terminal
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from Data.Datasets import PetaleRFDataset
from Data.Sampling import RandomStratifiedSampler
from json import dump
from SQL.constants import *
from SQL.DataManagement.Utils import initialize_petale_data_manager
from SQL.DataManagement.Helpers import retrieve_numerical


def argument_parser():
    """
    This function defines a parser to enable user to generate masks from a learning table
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python generate_masks.py [table] [target]',
                                     description="This program enables to generate stratified masks"
                                                 " associated to a learning table.")

    parser.add_argument('-tab', '--table', type=str, help="Name of the learning table")

    parser.add_argument('-tc', '--target_column', type=str, help="Name of the column to use as target")

    parser.add_argument('-s', '--seed', type=int, default=SEED,
                        help=f"Seed value used to create masks (default = {SEED})")

    parser.add_argument('-vsize', '--validation_size', type=float, default=0.20,
                        help="Percentage of data to use as valid set (default = 0.20)")

    parser.add_argument('-tsize', '--test_size', type=float, default=0.20,
                        help="Percentage of data to use as test set (default = 0.20)")

    parser.add_argument('-fn', '--file_name', type=str, help="Name of the json file to store the masks")

    parser.add_argument('-a', '--alpha', type=float, default=SAMPLING_OUTLIER_ALPHA,
                        help=f"IQR multiplier used to check numerical variable range validity"
                             f" of the test and valid masks (default = {SAMPLING_OUTLIER_ALPHA})")

    parser.add_argument('-k', '--nb_out_split', type=int, default=20,
                        help="Number of outer splits masks to produce (default = 20)")

    parser.add_argument('-l', '--nb_in_split', type=int, default=20,
                        help="Number of inner splits masks to produce (default = 20)")

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


if __name__ == '__main__':

    # We retrieve parser's arguments
    args = argument_parser()

    # We initialize a data manager
    data_manager = initialize_petale_data_manager()

    # We retrieve the table needed
    df = data_manager.get_table(args.table)

    # We extract an holdout set from the whole dataframe using a sampler
    cont_cols = list(retrieve_numerical(df, []).columns.values)
    cat_cols = [c for c in df.columns.values if c not in [PARTICIPANT, args.target_column] + cont_cols]
    dataset = PetaleRFDataset(df, args.target_column, cont_cols=cont_cols, cat_cols=cat_cols)
    rss = RandomStratifiedSampler(dataset, n_out_split=args.nb_out_split, n_in_split=args.nb_in_split,
                                  valid_size=args.validation_size, test_size=args.test_size,
                                  random_state=args.seed, alpha=args.alpha, patience=1000)
    masks = rss()

    # We dump the masks in a json file
    with open(f"{args.file_name}.json", "w") as file:
        dump(masks, file, indent=True)
