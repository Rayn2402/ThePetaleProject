"""
Author : Nicolas Raymond

This file contains the procedure to create the table associated to DEX and DOX cumulative doses.
"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from pandas import read_csv
from constants import DEX_DOX_TABLE
import os


COL = {"Participant": "text", "DEX (mg/m2)": "numeric", "DOX (mg/m2)": "numeric"}
DIR = "csv_files"
EXT = "csv"
PATH = os.path.join(DIR, f"{DEX_DOX_TABLE}.{EXT}")

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    df = read_csv(PATH, sep=";")

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, DEX_DOX_TABLE, COL)
