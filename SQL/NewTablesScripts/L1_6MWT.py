"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain "Learning_1 General" table in the database.
This table will consist of one of the simplest dataset that we will use to train our different model.
"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from constants import *
from SQL.NewTablesScripts.L0_WARMUP import get_missing_update
import pandas as pd

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We retrieve the tables needed
    six_df = data_manager.get_table(SIXMWT)
    gen_df = data_manager.get_table(GENERALS)

    # We remove VO2R_MAX from gen_df
    gen_df = gen_df.drop([VO2R_MAX], axis=1)

    # We remove survivors that as null fitness_lvl value
    gen_df = gen_df[~(gen_df[FITNESS_LVL].isnull())]

    # We concatenate the tables
    completed_df = pd.merge(gen_df, six_df, on=[PARTICIPANT], how=INNER)

    # We look quickly at the missing values
    get_missing_update(completed_df)        # 84 missing values out of 3717 (2.6%)

    # We create the dictionary needed to create the table
    types = {}
    for col in completed_df.columns:
        types[col] = TYPES[col]

    types.pop(FITNESS_LVL)
    types[FITNESS_LVL] = TYPES[FITNESS_LVL]

    # We create the table
    data_manager.create_and_fill_table(completed_df, LEARNING_1, types, primary_key=[PARTICIPANT])