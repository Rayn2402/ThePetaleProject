
"""
Author : Nicolas Raymond

This file contains the procedure to create the table PANDORA containing the IDs conversion map
to match patient with genomic results.
"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from pandas import read_csv
from SQL.NewTablesScripts.constants import PARTICIPANT, REF_NAME, TYPES, PETALE_PANDORA
import os


COL = {PARTICIPANT: TYPES[PARTICIPANT], REF_NAME: TYPES[REF_NAME]}
DIR = "csv_files"
EXT = "csv"
SEP = ","

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    df = read_csv(os.path.join(DIR, f"{PETALE_PANDORA}.{EXT}"), sep=SEP)

    # We modify data in the "Reference name" column to only keep the part after the last "-"
    df[REF_NAME] = df[REF_NAME].apply(lambda x: x.split("-")[-1].lstrip('0'))

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, PETALE_PANDORA, COL, primary_key=[PARTICIPANT])
