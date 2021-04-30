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
from SQL.NewTablesScripts.constants import ID_TABLE, PARTICIPANT, TYPES
import os

DIR = "csv_files"
EXT = "csv"
PATH = os.path.join(DIR, f"{ID_TABLE}.{EXT}")

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    IDs = read_csv(PATH)
    IDs[PARTICIPANT] = IDs[PARTICIPANT].astype('string').apply(fill_id)

    # We create and fill the table in the database
    data_manager.create_and_fill_table(IDs, ID_TABLE,
                                       {PARTICIPANT: TYPES[PARTICIPANT]},
                                       primary_key=[PARTICIPANT])
