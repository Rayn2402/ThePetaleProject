
"""
Author : Nicolas Raymond

This file contains the procedure to create the table PANDORA containing the IDs conversion map
to match patient with genomic results.
"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from pandas import read_csv
import os


TABLE_NAME = "PETALE_PANDORA"
COL = {"ID": "text", "Reference name": "text"}
DIR = "csv_files"
EXT = "csv"
PATH = os.path.join(DIR, f"{TABLE_NAME}.{EXT}")
SEP = ","

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    df = read_csv(PATH, sep=SEP)

    # We modify data in the "Reference name" column to only keep the part after the last "-"
    df["Reference name"] = df["Reference name"].apply(lambda x: x.split("-")[-1].lstrip('0'))

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, TABLE_NAME, COL)
