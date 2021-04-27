"""
Author : Nicolas Raymond

This file stores the procedure to execute in order to obtain "FIXED_FEATURES_AND_COMPLICATIONS_(FILTERED)"
table in the database.

 FIXED_FEATURES_AND_COMPLICATIONS_(FILTERED) contains :

    Features:
     - Same features as FIXED FEATURES but without patients that have missing VO2r_max values

    Complications:
    - Same complications as FIXED FEATURES + Fitness (0: No, 1: Yes)

"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from constants import *
from numpy import select, nan
from SQL.NewTablesScripts.L0_WARMUP import get_missing_update
import pandas as pd

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We save the variables needed from Cardio_0
    C0_vars = PKEY + [VO2_MAX, VO2_MAX_PRED]

    # We retrieve the tables with fixed features, cardio variables and IDs with good VO2r MAX values
    df_fixed = data_manager.get_table(FIXED_FEATURES_AND_COMPLICATIONS)
    df_cardio_0 = data_manager.get_table(CARDIO_0, C0_vars)
    IDs = data_manager.get_table(ID_TABLE)

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
    complete_df = pd.merge(df_fixed, IDs, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_0, on=[PARTICIPANT], how=INNER)

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)  # 6 values out of 216X14 = 3024 (~0.2%)

    """ TABLE CREATION """
    types = {c: TYPES[c] for c in complete_df.columns}

    # We create the table
    data_manager.create_and_fill_table(complete_df, FIXED_FEATURES_AND_COMPLICATIONS_FILTERED,
                                       types=types, primary_key=[PARTICIPANT])




