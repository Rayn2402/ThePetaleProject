"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain "Learning_0_WarmUp" table.
This table will consist of one of the dataset two reproduce 6MWT experiment with a more complex model.
"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from SQL.DataManager.Helpers import AbsTimeLapse
from constants import *
import pandas as pd


def get_missing_update(df):
    """
    Prints the number of rows and the number of missing values for each column
    :param df: pandas dataframe
    """
    print("Current number of rows : ", df.shape[0])
    print("Missing counts : ")
    print(df.isnull().sum(axis=0), "\n\n")


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

    # We create the dictionary needed to create the table
    types = {}
    for var in all_vars:
        types[var] = TYPES[var]

    # We make sure that the target is at the end
    types.pop(VO2R_MAX)
    types[VO2R_MAX] = TYPES[VO2R_MAX]

    # We create the table
    data_manager.create_and_fill_table(complete_df, LEARNING_0, types, primary_key=[PARTICIPANT])


