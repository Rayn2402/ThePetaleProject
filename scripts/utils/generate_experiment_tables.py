"""
Filename: generate_experiment_tables.py

Author: Nicolas Raymond

Description: Script to create a learning set and a holdout set from a dataset.

Date of last modification: 2022/07/13
"""
import argparse
import sys
from os.path import dirname, join, realpath
from pandas import read_csv

# Imports specific to project
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from settings.paths import Paths
from src.data.extraction.constants import *
from src.data.extraction.data_management import PetaleDataManager
from src.data.extraction.helpers import retrieve_numerical_var
from src.data.processing.datasets import MaskType, PetaleDataset
from src.data.processing.sampling import RandomStratifiedSampler
from src.utils.argparsers import print_arguments


def argument_parser():
    """
    Creates a parser to extract a learning set and an holdout set from a dataset
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python generate_experiment_tables.py [...]',
                                     description="Extracts a learning set and an holdout set from a dataset."
                                                 "Can either work with a dataset from a postgresql database"
                                                 "or a csv")

    parser.add_argument('-csv', '--csv', type=str,
                        help='Path of the csv file containing the dataset')

    parser.add_argument('-rt', '--raw_table', type=str,
                        help="Name of the raw dataset (ex. 'VO2_DATASET')")

    parser.add_argument('-nt', '--new_table', type=str,
                        help="Name of the new table created")

    parser.add_argument('-ocsv', '--outliers_csv', type=str,
                        help="Path of the csv file containing participant ids to remove")

    parser.add_argument('-sep', '--csv_separator', type=str, default=",",
                        help="Separation character used in the csv files (default = ',')")

    parser.add_argument('-tc', '--target_column', type=str,
                        help="Name of the column to use as target")

    parser.add_argument('-cat', '--categorical', default=False, action='store_true',
                        help='True if the target is categorical')

    parser.add_argument('-s', '--seed', type=int, default=SEED,
                        help=f"Seed value used to create holdout set (default = {SEED})")

    parser.add_argument('-hs', '--holdout_size', type=float, default=0.10,
                        help="Percentage of data to use as holdout set (default = 0.1)")

    parser.add_argument('-a', '--alpha', type=float, default=SAMPLING_OUTLIER_ALPHA,
                        help=f"IQR multiplier used to check numerical variable range"
                             f" validity of the holdout created (default = {SAMPLING_OUTLIER_ALPHA})")

    arguments = parser.parse_args()

    # Print arguments
    print_arguments(arguments)

    return arguments


if __name__ == '__main__':

    # We retrieve parser's arguments
    args = argument_parser()

    # Validation of arguments
    if args.csv is None and args.raw_table is None:
        raise ValueError('A csv or a table name must be provided')

    if args.csv is not None:

        # We load the dataframe from the csv
        df = read_csv(args.csv, sep=args.csv_separator)

    else:

        # We build a data manager to interact with the database
        data_manager = PetaleDataManager()

        # We retrieve the raw table needed
        df = data_manager.get_table(args.raw_table)

    if args.outliers_csv is not None:

        # We retrieve participant ids to remove
        outliers_ids = read_csv(args.outliers_csv, sep=args.csv_separator)
        outliers_ids[PARTICIPANT] = outliers_ids[PARTICIPANT].astype(str).apply(PetaleDataManager.fill_participant_id)

        # We remove the ids
        df = df.loc[~df[PARTICIPANT].isin(list(outliers_ids[PARTICIPANT].values)), :]

    # We identify numerical and categorical columns
    cont_cols = [c for c in list(retrieve_numerical_var(df, []).columns.values) if c != args.target_column]
    cat_cols = [c for c in df.columns.values if c not in [PARTICIPANT, args.target_column] + cont_cols]

    # We create a temporary dataset
    dataset = PetaleDataset(df,
                            args.target_column,
                            cont_cols=cont_cols,
                            cat_cols=cat_cols,
                            classification=args.categorical)

    # We extract an holdout set from the dataset using random stratified sampling
    rss = RandomStratifiedSampler(dataset,
                                  n_out_split=1,
                                  n_in_split=0,
                                  valid_size=0,
                                  test_size=args.holdout_size,
                                  random_state=args.seed,
                                  alpha=args.alpha)
    masks = rss()
    learning_idx, hold_out_idx = masks[0][MaskType.TRAIN], masks[0][MaskType.TEST]
    learning_df, hold_out_df = df.iloc[learning_idx, :], df.iloc[hold_out_idx, :]

    if args.csv is not None:

        # We save the dataframes into csv files
        directory = join(Paths.DATA, args.new_table)
        learning_df.to_csv(f"{directory}_LEARNING_SET.csv")
        hold_out_df.to_csv(f"{directory}_HOLDOUT_SET.csv")

    else:

        # We create the dictionary needed to create the postgresql tables
        types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(learning_df.columns)}

        # We create the tables
        data_manager.create_and_fill_table(learning_df, f"{args.new_table}_LEARNING_SET", types, primary_key=[PARTICIPANT])
        data_manager.create_and_fill_table(hold_out_df, f"{args.new_table}_HOLDOUT_SET", types, primary_key=[PARTICIPANT])
        print("\nTables created!")
