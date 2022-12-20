"""
Filename: GENERALS.py

Authors: Nicolas Raymond

Description: This file stores the procedure to execute in
             order to obtain "GENERALS" table in the database.

             GENERALS contains :

            Features:
             - SEX
             - AGE AT DIAGNOSIS
             - DURATION OF TREATMENT (DT)
             - RADIOTHERAPY DOSE
             - DOX DOSE
             - DEX (0; >0, <=Med; >Med) where Med is the median without 0's
             - GESTATIONAL AGE AT BIRTH (<37w, >=37w)
             - WEIGHT AT BIRTH (<2500g, >=2500g)
             - HEIGHT
             - WEIGHT
             - SMOKING (0: No, 1: Yes)
             - AGE
             - TSEOT
             - MVPLA
             - TAS_REST
             - TAD_REST

             Outcomes:
             - V02R_MAX
             - FITNESS COMPLICATIONS (0: No, 1: Yes)

Date of last modification : 2021/11/05
"""
import pandas as pd
import sys

from os.path import realpath, dirname

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.extraction.helpers import get_abs_years_timelapse, get_missing_update

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager()

    # We save the variables needed from BASELINE_FEATURES_AND_COMPLICATIONS_PLUS_FITNESS
    BASE_vars = [FITNESS_COMPLICATIONS, PARTICIPANT, SEX,
                 AGE_AT_DIAGNOSIS, DT, RADIOTHERAPY_DOSE,
                 DOX, DEX, BIRTH_AGE, BIRTH_WEIGHT]

    # We save the variables needed from General_1
    G1_vars = PKEY + [DATE, DATE_OF_BIRTH, HEIGHT, WEIGHT, SMOKING]

    # We save the variables needed from General_2
    G2_vars = PKEY + [DATE, DATE_OF_TREATMENT_END]

    # We save the variables needed from Cardio_0
    C0_vars = PKEY + [TAS_REST, TAD_REST, VO2R_MAX]

    # We save the variables needed from Cardio_3
    C3_vars = PKEY + [MVLPA]

    # We retrieve the tables with the variables
    df_base = data_manager.get_table(BASE_FEATURES_AND_COMPLICATIONS_PLUS_FITNESS, BASE_vars)
    df_general_1 = data_manager.get_table(GEN_1, G1_vars)
    df_general_2 = data_manager.get_table(GEN_2, G2_vars)
    df_cardio_0 = data_manager.get_table(CARDIO_0, C0_vars)
    df_cardio_3 = data_manager.get_table(CARDIO_3, C3_vars)

    # We only keep survivors from Phase 1
    df_general_1 = df_general_1[df_general_1[TAG] == PHASE]
    df_general_2 = df_general_2[df_general_2[TAG] == PHASE]
    df_cardio_0 = df_cardio_0[df_cardio_0[TAG] == PHASE]
    df_cardio_3 = df_cardio_3[df_cardio_3[TAG] == PHASE]

    """ GENERAL 1 DATA HANDLING """
    # We add the "Age" column
    get_abs_years_timelapse(df_general_1, AGE, DATE_OF_BIRTH, DATE)

    # We remove unnecessary variable
    df_general_1 = df_general_1.drop([DATE_OF_BIRTH, DATE], axis=1)

    """ GENERAL 2 DATA HANDLING """
    # We add a new column "Time since end of treatment" (TSEOT) (years)
    get_abs_years_timelapse(df_general_2, TSEOT, DATE_OF_TREATMENT_END, DATE)

    # We delete unnecessary variables from the dataframe
    df_general_2 = df_general_2.drop([DATE_OF_TREATMENT_END, DATE], axis=1)

    """ DATAFRAME CONCATENATION """
    # We concatenate all the dataframes
    complete_df = pd.merge(df_base, df_general_1, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_general_2, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_3, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_0, on=PKEY, how=INNER)

    # We remove TAG variable
    complete_df = complete_df.drop([TAG], axis=1)

    # We reorder table columns and put "Fitness complications?" at the end
    cols = list(complete_df.columns.values)
    complete_df = complete_df[cols[1:]+cols[:1]]

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)

    """ TABLE CREATION """
    types = {c: TYPES[c] for c in complete_df.columns}

    # We create the table
    data_manager.create_and_fill_table(complete_df, GENERALS, types=types, primary_key=[PARTICIPANT])
