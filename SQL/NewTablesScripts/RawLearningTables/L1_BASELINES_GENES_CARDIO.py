
"""
Author : Nicolas Raymond

This file contains the procedure to execute in order to obtain the following tables :
- L1_BASELINES_CARDIOMETABOLIC_RAW
- L1_BASELINES_GENES_CARDIOMETABOLIC_RAW
- L1_BASELINES_ALLGENES_CARDIOMETABOLIC_RAW
"""

from SQL.DataManagement.Utils import initialize_petale_data_manager
from SQL.constants import *
from SQL.DataManagement.Helpers import get_missing_update
from Data.Cleaning import DataCleaner
from os.path import join
import pandas as pd


if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We save the variables needed from BASELINES and COMPLICATIONS table
    base_vars = [PARTICIPANT, SEX, AGE_AT_DIAGNOSIS, DT, RADIOTHERAPY_DOSE,
                 DOX, DEX, BIRTH_AGE, BIRTH_WEIGHT, CARDIOMETABOLIC_COMPLICATIONS]

    # We retrieve the tables needed
    baselines_df = data_manager.get_table(BASE_FEATURES_AND_COMPLICATIONS, base_vars)

    for record, genes, new_table in [("BASELINES_CARDIO", None, LEARNING_1),
                                     ("BASELINES_GENES_CARDIO", GENES_12, LEARNING_1_1),
                                     ("BASELINES_ALLGENES_CARDIO", ALLGENES, LEARNING_1_2)]:

        # Initialization of data cleaner
        data_cleaner = DataCleaner(join(CLEANING_RECORDS, record), column_thresh=COLUMN_REMOVAL_THRESHOLD,
                                   row_thresh=ROW_REMOVAL_THRESHOLD, outlier_alpha=OUTLIER_ALPHA)

        if genes is not None:
            genes_df = data_manager.get_table(genes)

            # We concatenate the tables
            complete_df = pd.merge(genes_df, baselines_df, on=[PARTICIPANT], how=INNER)
        else:
            complete_df = baselines_df

        # We look at the missing data
        get_missing_update(complete_df)

        # We remove rows and columns with too many missing values and stores other cleaning suggestions
        complete_df = data_cleaner(complete_df)

        # We look at the missing data
        get_missing_update(complete_df)

        # We create the dictionary needed to create the table
        types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(complete_df.columns)}
    
        # We create the RAW learning table
        data_manager.create_and_fill_table(complete_df, f"{new_table}_{RAW}", types, primary_key=[PARTICIPANT])
