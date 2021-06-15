"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain official learning table
from a RAW table
"""

from pandas import read_csv
from src.data.extraction.constants import *
from src.data.extraction.data_management import PetaleDataManager
from src.data.extraction.helpers import fill_id, retrieve_numerical
from src.data.processing.datasets import PetaleRFDataset
from src.data.processing.sampling import RandomStratifiedSampler

import argparse


def argument_parser():
    """
    This function defines a parser that enables a user to create a learning
    set from a raw table
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python generate_learning_tables.py [raw_table] [new_table]'
                                           ' [target_column] [outliers_csv]',
                                     description="This program enables to remove participants from raw"
                                                 " learning table and extract an holdout set.")

    parser.add_argument('-u', '--user', type=str,
                        help="Valid username to access petale database")

    parser.add_argument('-rt', '--raw_table', type=str,
                        help="Name of the raw learning table (ex. 'L0_WARMUP_RAW')")

    parser.add_argument('-nt', '--new_table', type=str,
                        help="Name of the new table created")

    parser.add_argument('-ocsv', '--outliers_csv', type=str,
                        help="Path of the csv file containing participant ids to remove")

    parser.add_argument('-sep', '--csv_separator', type=str, default=",",
                        help="Separation character used in the csv file (default = ',')")

    parser.add_argument('-tc', '--target_column', type=str,
                        help="Name of the column to use as target")

    parser.add_argument('-s', '--seed', type=int, default=SEED,
                        help=f"Seed value used to create holdout set (default = {SEED})")

    parser.add_argument('-hs', '--holdout_size', type=float, default=0.10,
                        help="Percentage of data to use as holdout set (default = 0.1)")

    parser.add_argument('-a', '--alpha', type=float, default=SAMPLING_OUTLIER_ALPHA,
                        help=f"IQR multiplier used to check numerical variable range"
                             f" validity of the holdout created (default = {SAMPLING_OUTLIER_ALPHA})")

    args = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("\n")

    return args


if __name__ == '__main__':

    # We retrieve parser's arguments
    args = argument_parser()

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager(args.user)

    # We retrieve the raw table needed
    df = data_manager.get_table(args.raw_table)

    # We retrieve participant ids to remove
    outliers_ids = read_csv(args.outliers_csv, sep=args.csv_separator)
    outliers_ids[PARTICIPANT] = outliers_ids[PARTICIPANT].astype(str).apply(fill_id)

    # We remove the ids
    df = df.loc[~df[PARTICIPANT].isin(list(outliers_ids[PARTICIPANT].values)), :]

    # We extract an holdout set from the whole dataframe using a sampler
    cont_cols = list(retrieve_numerical(df, []).columns.values)
    cat_cols = [c for c in df.columns.values if c not in [PARTICIPANT, args.target_column] + cont_cols]
    dataset = PetaleRFDataset(df, args.target_column, cont_cols=cont_cols, cat_cols=cat_cols)
    rss = RandomStratifiedSampler(dataset, n_out_split=1, n_in_split=0,
                                  valid_size=0, test_size=args.holdout_size,
                                  random_state=args.seed, alpha=args.alpha)
    masks = rss()
    learning_idx, hold_out_idx = masks[0]["train"], masks[0]["test"]
    learning_df, hold_out_df = df.iloc[learning_idx, :], df.iloc[hold_out_idx, :]

    # We create the dictionary needed to create the table
    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(learning_df.columns)}

    # We create the tables
    data_manager.create_and_fill_table(learning_df, args.new_table, types, primary_key=[PARTICIPANT])
    data_manager.create_and_fill_table(hold_out_df, f"{args.new_table}_HOLDOUT", types, primary_key=[PARTICIPANT])
    print("\nTables created!")
