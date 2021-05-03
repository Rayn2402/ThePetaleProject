"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain "Learning_1 General" table in the database.
This table will consist of one of the simplest dataset that we will use to train our different model.
"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from SQL.NewTablesScripts.L0_WARMUP import get_missing_update
from Data.Sampling import split_train_test
from SQL.NewTablesScripts.constants import *
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
    gen_df = gen_df[~(gen_df[FITNESS_COMPLICATIONS].isnull())]

    # We concatenate the tables
    complete_df = pd.merge(gen_df, six_df, on=[PARTICIPANT], how=INNER)

    # We look quickly at the missing values
    get_missing_update(complete_df)        # 84 missing values out of 3717 (2.6%)

    # We extract an holdout set from the complete df
    learning_df, hold_out_df = split_train_test(complete_df, FITNESS_COMPLICATIONS, test_size=0.10, random_state=SEED)

    # We create the dictionary needed to create the table
    types = {c: TYPES[c] for c in complete_df.columns}
    types.pop(FITNESS_COMPLICATIONS)
    types[FITNESS_COMPLICATIONS] = TYPES[FITNESS_COMPLICATIONS]

    # We create the table
    data_manager.create_and_fill_table(learning_df, LEARNING_1, types, primary_key=[PARTICIPANT])
    data_manager.create_and_fill_table(hold_out_df, LEARNING_1_HOLDOUT, types, primary_key=[PARTICIPANT])
