"""
Filename: BASELINE_FEATURES_PLUS_FIT.py

Authors: Nicolas Raymond

Description: This file stores the procedure to execute in order to obtain
             "BASE_FEATURES_AND_COMPLICATIONS_+_FITNESS" table in the database.

     BASE_FEATURES_AND_COMPLICATIONS contains :

        Features:
         - Same features as BASE_FEATURES_AND_COMPLICATIONS but without patients
           that have missing VO2r_max values

        Complications:
        - Same complications as BASE_FEATURES_AND_COMPLICATIONS + Fitness (0: No, 1: Yes)

Date of last modification : 2021/11/05
"""
import pandas as pd
import sys

from os.path import dirname, realpath
from numpy import select

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.extraction.helpers import get_missing_update

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager()

    # We save the variables needed from Cardio_0
    C0_vars = PKEY + [VO2_MAX, VO2_MAX_PRED]

    # We retrieve the tables with fixed features, cardio variables and IDs with good VO2r MAX values
    df_fixed = data_manager.get_table(BASE_FEATURES_AND_COMPLICATIONS)
    df_cardio_0 = data_manager.get_table(CARDIO_0, C0_vars)
    VO2_ids = data_manager.get_table(VO2_ID_TABLE)

    # We only keep survivors from Phase 1 in the cardio table
    df_cardio_0 = df_cardio_0[df_cardio_0[TAG] == PHASE]

    # We remove TAG variable
    df_cardio_0 = df_cardio_0.drop([TAG], axis=1)

    """ CARDIO 0 DATA HANDLING """
    # We create a new column temporary called "fitness"
    df_cardio_0[FITNESS] = df_cardio_0[VO2_MAX] / df_cardio_0[VO2_MAX_PRED]

    # We create a new column fit "Fitness Complications?" based on fitness
    conditions = [(df_cardio_0[FITNESS] >= 1),
                  (df_cardio_0[FITNESS] < 1),
                  (pd.isnull(df_cardio_0[FITNESS]))]

    binaries = [0, 1, nan]
    df_cardio_0[FITNESS_COMPLICATIONS] = select(conditions, binaries)

    # We delete unnecessary variables from the dataframe
    df_cardio_0 = df_cardio_0.drop([VO2_MAX, VO2_MAX_PRED, FITNESS], axis=1)

    """ DATAFRAME CONCATENATION """
    # We concatenate all the dataframes
    complete_df = pd.merge(df_fixed, VO2_ids, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_0, on=[PARTICIPANT], how=INNER)

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)  # 29 values out of 216X13 = 2808 (1%)

    """ TABLE CREATION """
    types = {c: TYPES[c] for c in complete_df.columns}

    # We create the table
    data_manager.create_and_fill_table(complete_df, BASE_FEATURES_AND_COMPLICATIONS_PLUS_FITNESS,
                                       types=types, primary_key=[PARTICIPANT])




