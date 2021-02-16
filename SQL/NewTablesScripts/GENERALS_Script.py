"""
Author : Nicolas Raymond

This file stores the procedure to execute in order to obtain "GENERALS" table in the database.

"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from SQL.DataManager.Helpers import AbsTimeLapse
from constants import *
from numpy import select, nan
from SQL.NewTablesScripts.Learning_0_Script import get_missing_update
import pandas as pd

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We save the variables needed from General_1
    G1_vars = PKEY + [DATE, DATE_OF_BIRTH, SEX, HEIGHT, WEIGHT, SMOKING]

    # We save the variables needed from General_2
    G2_vars = PKEY + [DATE, AGE_AT_DIAGNOSIS, DATE_OF_TREATMENT_END, DATE_OF_DIAGNOSIS, RADIOTHERAPY, RADIOTHERAPY_DOSE]

    # We save the variables needed from Cardio_0
    C0_vars = PKEY + [VO2R_MAX, VO2_MAX, VO2_MAX_PRED, TAS_REST, TAD_REST]

    # We save the variables needed from Cardio_3
    C3_vars = PKEY + [QAPL8]

    # We save variables needed from DEX_DOX
    DEX_DOX_vars = [PARTICIPANT, DEX, DOX]

    # We save a set with all the variables and initialize two new lists for removed and added columns
    all_vars = set(G1_vars+G2_vars+C0_vars+C3_vars+DEX_DOX_vars)
    new_columns = []
    deleted_columns = []

    # We retrieve the tables with variables and the table with the filtered ID
    df_general_1 = data_manager.get_table(GEN_1, G1_vars)
    df_general_2 = data_manager.get_table(GEN_2, G2_vars)
    df_cardio_0 = data_manager.get_table(CARDIO_0, C0_vars)
    df_cardio_3 = data_manager.get_table(CARDIO_3, C3_vars)
    DEX_DOX = data_manager.get_table(DEX_DOX_TABLE, DEX_DOX_vars)
    IDs = data_manager.get_table(ID_TABLE)

    # We only keep survivors from Phase 1
    df_general_1 = df_general_1[df_general_1[TAG] == PHASE]
    df_general_2 = df_general_2[df_general_2[TAG] == PHASE]
    df_cardio_0 = df_cardio_0[df_cardio_0[TAG] == PHASE]
    df_cardio_3 = df_cardio_3[df_cardio_3[TAG] == PHASE]

    """ CARDIO 0 DATA HANDLING """
    # We create a new column temporary called "fitness"
    df_cardio_0[FITNESS] = df_cardio_0[VO2_MAX] / df_cardio_0[VO2_MAX_PRED]

    # We create a new column fit "fitness lvls" based on fitness
    conditions = [(df_cardio_0[FITNESS] < 0.85),
                  (df_cardio_0[FITNESS] >= 0.85) & (df_cardio_0[FITNESS] < 1),
                  (df_cardio_0[FITNESS] > 1),
                  (pd.isnull(df_cardio_0[FITNESS]))]

    lvls = [0, 1, 2, nan]
    df_cardio_0[FITNESS_LVL] = select(conditions, lvls)
    new_columns.append(FITNESS_LVL)

    # We delete unnecessary variables from the dataframe
    df_cardio_0 = df_cardio_0.drop([VO2_MAX, VO2_MAX_PRED, FITNESS], axis=1)
    deleted_columns += [VO2_MAX, VO2_MAX_PRED]

    """ GENERAL 1 DATA HANDLING """
    # We add the "Age" column
    AbsTimeLapse(df_general_1, AGE, DATE_OF_BIRTH, DATE)
    new_columns.append(AGE)

    # We remove unnecessary variable
    df_general_1 = df_general_1.drop([DATE_OF_BIRTH, DATE], axis=1)
    deleted_columns += [DATE_OF_BIRTH, DATE]

    """ GENERAL 2 DATA HANDLING """
    # We add a new column "Duration of treatment" (DT) (years)
    AbsTimeLapse(df_general_2, DT, DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END)
    new_columns.append(DT)

    # We add a new column "Time since end of treatment" (TSEOT) (years)
    AbsTimeLapse(df_general_2, TSEOT, DATE_OF_TREATMENT_END, DATE)
    new_columns.append(TSEOT)

    # We adjust the column "Radiotherapy dose" because it is null if "Radiotherapy?" column equals 0
    df_general_2.loc[df_general_2[RADIOTHERAPY] == '0.0', RADIOTHERAPY_DOSE] = 0

    # We delete unnecessary variables from the dataframe
    df_general_2 = df_general_2.drop([RADIOTHERAPY, DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END, DATE], axis=1)
    deleted_columns += [RADIOTHERAPY, DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END]

    """ DEX_DOX DATA HANDLING """
    # We add a new categorical column based on DEX cumulative dose
    conditions = [(~DEX_DOX[DEX].isnull()), (DEX_DOX[DEX].isnull())]
    categories = ["Yes", "No or unknown"]

    DEX_DOX[DEX_PRESENCE] = select(conditions, categories)
    new_columns.append(DEX_PRESENCE)

    # We delete unnecessary variables from the dataframe
    DEX_DOX = DEX_DOX.drop([DEX], axis=1)
    deleted_columns.append(DEX)

    """ DATAFRAME CONCATENATION """
    # We concatenate all the dataframes
    complete_df = pd.merge(IDs, DEX_DOX, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_general_1, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_general_2, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_0, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_3, on=PKEY, how=INNER)

    # We remove TAG variable
    complete_df = complete_df.drop([TAG], axis=1)
    deleted_columns.append(TAG)

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)

    """ TABLE CREATION """
    # We delete non needed columns from list of vars
    for deleted_col in deleted_columns:
        all_vars.remove(deleted_col)

    # We add the new columns to the list of vars
    for new_col in new_columns:
        all_vars.add(new_col)

    # We create a dictionary with the remaining variables
    types = {PARTICIPANT: TYPES[PARTICIPANT]}  # We want the participant ID as the first column
    all_vars.remove(PARTICIPANT)
    all_vars.remove(VO2R_MAX)
    all_vars.remove(FITNESS_LVL)

    for var in all_vars:
        types[var] = TYPES[var]

    # We want the targets as the last columns
    types[FITNESS_LVL] = TYPES[FITNESS_LVL]
    types[VO2R_MAX] = TYPES[VO2R_MAX]

    # We create the table
    data_manager.create_and_fill_table(complete_df, GENERALS, types=types, primary_key=[PARTICIPANT])