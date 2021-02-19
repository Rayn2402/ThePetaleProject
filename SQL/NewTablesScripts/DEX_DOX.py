"""
Author : Nicolas Raymond

This file contains the procedure to create the table associated to DEX and DOX cumulative doses.
"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from pandas import read_csv
import os


TABLE_NAME = "DEX_DOX"
COL = {"Participant": "text", "DEX (mg/m2)": "numeric", "DOX (mg/m2)": "numeric"}
DIR = "csv_files"
EXT = "csv"
PATH = os.path.join(DIR, f"{TABLE_NAME}.{EXT}")

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    df = read_csv(PATH, sep=";")

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, TABLE_NAME, COL)
