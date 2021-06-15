"""
Author : Nicolas Raymond

This file contains the procedure to create the table with IDs of
invalid participant that didn't receive the same treatment as the others

"""
from os.path import join, realpath, dirname
from pandas import read_csv

import sys

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import INVALID_ID_TABLE, PARTICIPANT, TYPES
    from src.data.extraction.data_management import initialize_petale_data_manager
    from src.data.extraction.helpers import fill_id

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    invalid_ids = read_csv(join(Paths.CSV_FILES.value, f"{INVALID_ID_TABLE}.csv"))
    invalid_ids[PARTICIPANT] = invalid_ids[PARTICIPANT].astype('string').apply(fill_id)

    # We create and fill the table in the database
    data_manager.create_and_fill_table(invalid_ids, INVALID_ID_TABLE,
                                       {PARTICIPANT: TYPES[PARTICIPANT]}, primary_key=[PARTICIPANT])
