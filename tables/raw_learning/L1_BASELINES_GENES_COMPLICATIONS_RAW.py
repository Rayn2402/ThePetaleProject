"""
Filename: L1_BASELINES_GENES_COMPLICATIONS.py

Authors: Nicolas Raymond

Description: This file contains the procedure to execute in order
             to obtain "L1_BASELINES_GENES_COMPLICATIONS_RAW"

Date of last modification : 2021/11/05
"""
import pandas as pd
import sys

from os.path import join, realpath, dirname

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.extraction.helpers import get_missing_update
    from src.data.processing.cleaning import DataCleaner

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager()

    # We save the variables needed from BASELINES and COMPLICATIONS table
    base_vars = [PARTICIPANT, SEX, AGE_AT_DIAGNOSIS, DT, RADIOTHERAPY_DOSE,
                 DOX, DEX, BIRTH_AGE, BIRTH_WEIGHT, CARDIOMETABOLIC_COMPLICATIONS,
                 BONE_COMPLICATIONS, NEUROCOGNITIVE_COMPLICATIONS, COMPLICATIONS]

    # We retrieve the baselines variables needed
    baselines_df = data_manager.get_table(BASE_FEATURES_AND_COMPLICATIONS, base_vars)

    # We retrieve the complete genes table
    genes_df = data_manager.get_table(ALLGENES)

    # Initialization of data cleaner
    data_cleaner = DataCleaner(join(Paths.CLEANING_RECORDS, "BASELINES_ALLGENES_COMPLICATIONS"),
                               column_thresh=COLUMN_REMOVAL_THRESHOLD, row_thresh=ROW_REMOVAL_THRESHOLD,
                               outlier_alpha=OUTLIER_ALPHA, min_n_per_cat=MIN_N_PER_CAT,
                               max_cat_percentage=MAX_CAT_PERCENTAGE)
    # We concatenate the tables
    print(f"Removed participant : "
          f"{[p for p in list(baselines_df[PARTICIPANT].values) if p not in list(genes_df[PARTICIPANT].values)]}")
    complete_df = pd.merge(genes_df, baselines_df, on=[PARTICIPANT], how=INNER)

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We remove rows and columns with too many missing values and stores other cleaning suggestions
    complete_df = data_cleaner(complete_df)

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We create the dictionary needed to create the table
    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(complete_df.columns)}

    # We create the RAW learning table
    data_manager.create_and_fill_table(complete_df, f"{LEARNING_1}_{RAW}", types, primary_key=[PARTICIPANT])
