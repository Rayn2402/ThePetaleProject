"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain
"L1_BASELINES_CARDIOMETABOLIC_RAW" table.

"""

from SQL.DataManagement.Utils import initialize_petale_data_manager
from SQL.constants import *
from SQL.DataManagement.Helpers import get_missing_update
from Data.Cleaning import DataCleaner
from os.path import join

data_cleaner = DataCleaner(join(CLEANING_RECORDS, "BASELINES_CARDIO"), column_thresh=COLUMN_REMOVAL_THRESHOLD,
                           row_thresh=ROW_REMOVAL_THRESHOLD, outlier_alpha=OUTLIER_ALPHA)

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We save the variables needed from BASELINES and COMPLICATIONS table
    vars = [PARTICIPANT, SEX, AGE_AT_DIAGNOSIS, DT, RADIOTHERAPY_DOSE,
            DOX, DEX, BIRTH_AGE, BIRTH_WEIGHT, CARDIOMETABOLIC_COMPLICATIONS]

    # We retrieve the tables needed
    df = data_manager.get_table(BASE_FEATURES_AND_COMPLICATIONS, vars)

    # We look at the missing data
    get_missing_update(df)

    # We remove rows and columns with too many missing values and stores other cleaning suggestions
    df = data_cleaner(df)

    # We look at the missing data
    get_missing_update(df)

    # We create the dictionary needed to create the table
    types = {c: TYPES[c] for c in list(df.columns)}

    # We create the RAW learning table
    data_manager.create_and_fill_table(df, f"{LEARNING_1}_{RAW}", types, primary_key=[PARTICIPANT])




