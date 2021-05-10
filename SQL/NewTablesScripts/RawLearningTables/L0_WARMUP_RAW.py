"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain "L0_WARMUP_RAW".

"""

from SQL.DataManagement.Utils import initialize_petale_data_manager
from SQL.constants import *
from SQL.DataManagement.Helpers import get_missing_update
from Data.Cleaning import DataCleaner
from os.path import join
import pandas as pd

data_cleaner = DataCleaner(join(CLEANING_RECORDS, "WARMUP"), column_thresh=COLUMN_REMOVAL_THRESHOLD,
                           row_thresh=ROW_REMOVAL_THRESHOLD, outlier_alpha=OUTLIER_ALPHA)

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We save the variables needed from GENERALS
    GEN_vars = [PARTICIPANT, AGE, WEIGHT, DT, MVLPA, VO2R_MAX]

    # We save the variables needed from 6MWT
    SIXMWT_vars = [PARTICIPANT, TDM6_HR_END, TDM6_DIST]

    # We save a set with all the variables
    all_vars = set(GEN_vars+SIXMWT_vars)

    # We retrieve the tables needed
    gen_df = data_manager.get_table(GENERALS, GEN_vars)
    six_df = data_manager.get_table(SIXMWT, SIXMWT_vars)

    # We remove survivors with missing VO2R_MAX values
    gen_df = gen_df[~(gen_df[VO2R_MAX].isnull())]

    # We proceed to table concatenation
    complete_df = pd.merge(gen_df, six_df, on=[PARTICIPANT], how=INNER)

    # We look at the missing data
    get_missing_update(complete_df)

    # We remove rows and columns with too many missing values and stores other cleaning suggestions
    complete_df = data_cleaner(complete_df)

    # We look at the missing data
    get_missing_update(complete_df)

    # We create the dictionary needed to create the table
    types = {c: TYPES[c] for c in all_vars}

    # We make sure that the target is at the end
    types.pop(VO2R_MAX)
    types[VO2R_MAX] = TYPES[VO2R_MAX]

    # We create the RAW learning table
    data_manager.create_and_fill_table(complete_df, f"{LEARNING_0}_{RAW}", types, primary_key=[PARTICIPANT])


