"""
Author : Nicolas Raymond

This file contains the procedure to create the table with IDs of
participant that satisfies criteria of maximal effort.

IDs were sent by Maxime Caru

Criterias can be found in :

"Maximal cardiopulmonary exercise testing in childhood acute
 lymphoblastic leukemia survivors exposed to chemotherapy"

"""
from SQL.DataManager.Utils import initialize_petale_data_manager
from SQL.DataManager.Helpers import fill_id
from pandas import read_csv
import os

TABLE_NAME = "VO2_ID"
COL = "Participant"
DIR = "csv_files"
EXT = "csv"
PATH = os.path.join(DIR, f"{TABLE_NAME}.{EXT}")

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    IDs = read_csv(PATH)
    IDs[COL] = IDs[COL].astype('string').apply(fill_id)

    # We create and fill the table in the database
    data_manager.create_and_fill_table(IDs, TABLE_NAME, {COL: "text"})
