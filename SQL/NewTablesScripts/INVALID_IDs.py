"""
Author : Nicolas Raymond

This file contains the procedure to create the table with IDs of
invalid participant that didn't receive the same treatment as the others

"""
from SQL.DataManager.Utils import initialize_petale_data_manager
from SQL.DataManager.Helpers import fill_id
from pandas import read_csv
from SQL.NewTablesScripts.constants import INVALID_ID_TABLE, PARTICIPANT, TYPES
import os

DIR = "csv_files"
EXT = "csv"
PATH = os.path.join(DIR, f"{INVALID_ID_TABLE}.{EXT}")

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    invalid_ids = read_csv(PATH)
    invalid_ids[PARTICIPANT] = invalid_ids[PARTICIPANT].astype('string').apply(fill_id)

    # We create and fill the table in the database
    data_manager.create_and_fill_table(invalid_ids, INVALID_ID_TABLE,
                                       {PARTICIPANT: TYPES[PARTICIPANT]}, primary_key=[PARTICIPANT])
