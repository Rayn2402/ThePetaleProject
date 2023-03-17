"""
Filename: create_experiment_table.py

Authors: Nicolas Raymond

Description: This file is used to create the table containing the required data
             to realize the experiments regarding the prediction of VO2 peak

             The variables contained in the resulting table are:

             - PARTICIPANT ('Participant')
             - SEX ('34500 Sex')
             - VO2R_MAX ('35009 EE_VO2r_max')
             - WEIGHT ('34503 Weight')
             - TDM6_HR_END ('35149 TDM6_HR_6_2')
             - TDM6_DIST ('35142 TDM6_Distance_2')
             - DT ('Duration of treatment')
             - AGE ('Age')
             - MVLPA ('35116 QAPL8')

Date of last modification : --
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

    # We extract the required data from the existing tables
    df_cardio_0 = data_manager.get_table(CARDIO_0, PKEY + [VO2R_MAX])
    df_cardio_3 = data_manager.get_table(CARDIO_3, PKEY + [MVLPA])
    df_cardio_4 = data_manager.get_table(CARDIO_4, PKEY + [TDM6_DIST, TDM6_HR_END])
    df_general_1 = data_manager.get_table(GEN_1, PKEY + [DATE, DATE_OF_BIRTH, SEX, WEIGHT])
    df_general_2 = data_manager.get_table(GEN_2, PKEY + [DATE_OF_TREATMENT_END, DATE_OF_DIAGNOSIS])
    valid_ids = data_manager.get_table(VO2_ID_TABLE)
    invalid_ids = data_manager.get_table(INVALID_ID_TABLE)

    # We only keep survivors from Phase 1
    df_general_1 = df_general_1[df_general_1[TAG] == PHASE]

    """ GENERAL 1 DATA HANDLING """
    # We had 'Age' column
    get_abs_years_timelapse(df_general_1, AGE, DATE_OF_BIRTH, DATE)

    # We remove unnecessary variable
    df_general_1 = df_general_1.drop([DATE_OF_BIRTH, DATE], axis=1)

    """ GENERAL 2 DATA HANDLING """
    # We add a new column "Duration of treatment" (DT) (years)
    get_abs_years_timelapse(df_general_2, DT, DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END)

    # We delete unnecessary variables from the dataframe
    df_general_2 = df_general_2.drop([DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END], axis=1)

    """ DATAFRAME CONCATENATION """
    # We concatenate all the dataframes
    complete_df = pd.merge(valid_ids, df_general_1, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_general_2, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_0, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_3, on=PKEY, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_4, on=PKEY, how=INNER)

    # We remove TAG variable
    complete_df = complete_df.drop([TAG], axis=1)

    """ REMOVAL OF INVALID IDs """
    # We remove participant that were excluded from the PETALE study due to treatment history
    complete_df = complete_df[~(complete_df[PARTICIPANT].isin(list(invalid_ids[PARTICIPANT].values)))]

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)

    """ TABLE CREATION """
    types = {c: TYPES[c] for c in complete_df.columns}

    # We create the table
    data_manager.create_and_fill_table(complete_df, VO2_OFFICIAL_DATASET,
                                       types=types, primary_key=[PARTICIPANT])


















