"""
Filename: generate_masks.py

Author: Nicolas Raymond

Description: Script used to produce train, valid and test masks related to a dataframe

Date of last modification: 2022/07/27
"""
import argparse
import sys
from json import dump
from os.path import dirname, join, realpath
from pandas import read_csv

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from settings.paths import Paths
from src.data.processing.datasets import PetaleDataset
from src.data.processing.sampling import RandomStratifiedSampler
from src.data.extraction.constants import *
from src.data.extraction.data_management import PetaleDataManager
from src.data.extraction.helpers import retrieve_numerical_var
from src.utils.argparsers import print_arguments


def argument_parser():
    """
    Creates a parser for the masks generation
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python generate_masks.py -tab [table_name] -tc [target_column]'
                                           ' -fn [file_name] [...]',
                                     description="Generates stratified masks associated to a dataframe coming"
                                                 " from a postgresql table or a csv file.")

    parser.add_argument('-csv', '--csv', type=str,
                        help="Path of the csv file containing the dataset")

    parser.add_argument('-tab', '--table', type=str,
                        help="Name of the postgresql table")

    parser.add_argument('-tc', '--target_column', type=str,
                        help="Name of the column to use as target")

    parser.add_argument('-cat', '--categorical', default=False, action='store_true',
                        help='True if the target is categorical')

    parser.add_argument('-rc', '--removed_columns', nargs='*', type=str, default=[],
                        help="Columns to remove from the dataframe before creating the masks")

    parser.add_argument('-s', '--seed', type=int, default=SEED,
                        help=f"Seed value used to create masks (default = {SEED})")

    parser.add_argument('-vsize', '--validation_size', type=float, default=0.20,
                        help="Percentage of data used as valid set (default = 0.20)")

    parser.add_argument('-tsize', '--test_size', type=float, default=0.20,
                        help="Percentage of data used as test set (default = 0.20)")

    parser.add_argument('-fn', '--file_name', type=str,
                        help="Name of the json file used to store the masks")

    parser.add_argument('-a', '--alpha', type=float, default=SAMPLING_OUTLIER_ALPHA,
                        help=f"IQR multiplier used to check numerical variable range validity"
                             f" of the test and valid masks (default = {SAMPLING_OUTLIER_ALPHA})")

    parser.add_argument('-k', '--nb_out_split', type=int, default=10,
                        help="Number of outer splits masks to produce (default = 10)")

    parser.add_argument('-l', '--nb_in_split', type=int, default=10,
                        help="Number of inner splits masks to produce (default = 10)")

    arguments = parser.parse_args()

    # Print arguments
    print_arguments(arguments)

    return arguments


if __name__ == '__main__':

    # We retrieve arguments
    args = argument_parser()

    # Validation of arguments
    if args.csv is None and args.table is None:
        raise ValueError('A csv or a table name must be provided')

    if args.csv is None:

        # We initialize a data manager
        data_manager = PetaleDataManager()

        # We retrieve the table needed
        df = data_manager.get_table(args.table)

    else:

        # We retrieve the table needed
        df = read_csv(args.csv)

    # We remove unnecessary columns
    columns = [c for c in df.columns if c not in args.removed_columns]
    df = df[columns]

    # We identify numerical and categorical columns
    cont_cols = [c for c in list(retrieve_numerical_var(df, []).columns.values) if c != args.target_column]
    cat_cols = [c for c in df.columns.values if c not in [PARTICIPANT, args.target_column] + cont_cols]

    # We create a temporary dataset
    dataset = PetaleDataset(df, args.target_column,
                            cont_cols=cont_cols,
                            cat_cols=cat_cols,
                            classification=args.categorical)

    # We create stratified mask according to the target columns
    rss = RandomStratifiedSampler(dataset,
                                  n_out_split=args.nb_out_split,
                                  n_in_split=args.nb_in_split,
                                  valid_size=args.validation_size,
                                  test_size=args.test_size,
                                  random_state=args.seed,
                                  alpha=args.alpha,
                                  patience=1000)
    masks = rss()

    # We dump the masks in a json file
    with open(join(Paths.MASKS, f"{args.file_name}.json"), "w") as file:
        dump(masks, file, indent=True)
