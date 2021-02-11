"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain "Learning_0_WarmUp" table.
This table will consist of one of the dataset two reproduce 6MWT experiment with a more complex model.
"""

from SQL.DataManager.Utils import PetaleDataManager
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
    user_name = input("Enter your username to access PETALE database : ")
    data_manager = PetaleDataManager(user_name)

    # We save the variables needed from General_1
    G1_vars = [DATE] + PKEY + [DATE_OF_BIRTH, WEIGHT]

    # We save the variables needed from General_2
    G2_vars = PKEY + [DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END]

    # We save the variables needed from Cardio_0
    C0_vars = PKEY + [VO2R_MAX]

    # We save the variables needed from Cardio_3
    C3_vars = PKEY + [QAPL8]

    # We save the variables needed from Cardio_4
    C4_vars = PKEY + [TDM6_HR_END, TDM6_DIST]

    # We save a set with all the variables
    all_vars = set(G1_vars+G2_vars+C3_vars+C4_vars+C0_vars)

    # We retrieve the tables with variables and the table with the filtered ID
    df_general_1 = data_manager.get_table(GEN_1, G1_vars)
    df_general_2 = data_manager.get_table(GEN_2, G2_vars)
    df_cardio_0 = data_manager.get_table(CARDIO_0, C0_vars)
    df_cardio_3 = data_manager.get_table(CARDIO_3, C3_vars)
    df_cardio_4 = data_manager.get_table(CARDIO_4, C4_vars)
    IDs = data_manager.get_table(ID_TABLE)

    # We only keep survivors from Phase 1
    df_general_1 = df_general_1[df_general_1[TAG] == PHASE]
    df_general_2 = df_general_2[df_general_2[TAG] == PHASE]
    df_cardio_0 = df_cardio_0[df_cardio_0[TAG] == PHASE]
    df_cardio_3 = df_cardio_3[df_cardio_3[TAG] == PHASE]
    df_cardio_4 = df_cardio_4[df_cardio_4[TAG] == PHASE]

    # We remove survivors that have missing VO2r_max value
    df_cardio_0 = df_cardio_0[~df_cardio_0[VO2R_MAX].isnull()]

    # We add a new column Duration of treatment (years) to the table general_2
    AbsTimeLapse(df_general_2, DT, DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END)

    # We rename QALP8 as MVLPA
    df_cardio_3 = df_cardio_3.rename(columns={QAPL8: MVLPA})

    # We concatenate all the dataframes
    complete_df = pd.merge(IDs, df_general_1, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_general_2, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_0, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_3, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_4, on=PKEY, how=INNER)

    # We add the "Age" column
    AbsTimeLapse(complete_df, AGE, DATE_OF_BIRTH, DATE)

    # We remove the dates column and the "Tag" column
    deleted_columns = [DATE, DATE_OF_TREATMENT_END, DATE_OF_DIAGNOSIS, DATE_OF_BIRTH, TAG]
    complete_df = complete_df.drop(deleted_columns, axis=1)

    # We delete non needed columns from list of vars
    deleted_columns.append(QAPL8)
    for deleted_col in deleted_columns:
        all_vars.remove(deleted_col)

    # We add the new columns to the list of vars
    for new_col in [DT, MVLPA, AGE]:
        all_vars.add(new_col)

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)  # 19 continuous values are missing

    # We create a dictionary with the remaining variables
    types = {PARTICIPANT: TYPES[PARTICIPANT]}  # We want the participant ID as the first column

    all_vars.remove(PARTICIPANT)
    all_vars.remove(VO2R_MAX)

    for var in all_vars:
        types[var] = TYPES[var]

    types[VO2R_MAX] = TYPES[VO2R_MAX]  # We want the target as the last column

    # We filter the dataframe created
    complete_df = complete_df[types.keys()]

    # We create the table
    data_manager.create_and_fill_table(complete_df, LEARNING_0, types=types, primary_key=[PARTICIPANT])
