"""
Author : Nicolas Raymond

This file stores the procedure to execute in order to obtain "FIXED_FEATURES_AND_COMPLICATIONS"
table in the database.

 FIXED_FEATURES_AND_COMPLICATIONS contains :

    Features:
     - SEX
     - AGE AT DIAGNOSIS
     - DURATION OF TREATMENT (DT)
     - RADIOTHERAPY DOSE
     - DOX DOSE
     - DEX (0: "Unknown", 1: "Yes")
     - GESTATIONAL AGE AT BIRTH (1: <37w, 2: >=37w, 9: Unknown)
     - WEIGHT AT BIRTH (1: <2500g, 2: >=2500g, 3: Unknown)

    Complications:
    - Metabolic (0: No, 1: Yes)
    - Skeletal/Bone (0: No, 1: Yes)
    - Cardiac (0: No, 1: Yes)
    - Neurocognitive (0: No, 1: Yes)

"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from SQL.DataManager.Helpers import AbsTimeLapse
from constants import *
from numpy import select
from SQL.NewTablesScripts.L0_WARMUP import get_missing_update
import pandas as pd

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We save the variables needed from General_1
    G1_vars = PKEY + [SEX, BIRTH_AGE, BIRTH_WEIGHT]

    # We save the variables needed from General_2
    G2_vars = PKEY + [AGE_AT_DIAGNOSIS, DATE_OF_TREATMENT_END, DATE_OF_DIAGNOSIS, RADIOTHERAPY, RADIOTHERAPY_DOSE,
                      METABOLIC_COMPLICATIONS, BONE_COMPLICATIONS, CARDIAC_COMPLICATIONS, NEUROCOGNITIVE_COMPLICATIONS]

    # We save variables needed from DEX_DOX
    DEX_DOX_vars = [PARTICIPANT, DEX, DOX]

    # We retrieve the tables with variables and the table with IDs with good VO2r MAX values
    df_general_1 = data_manager.get_table(GEN_1, G1_vars)
    df_general_2 = data_manager.get_table(GEN_2, G2_vars)
    DEX_DOX = data_manager.get_table(DEX_DOX_TABLE, DEX_DOX_vars)

    # We only keep survivors from Phase 1
    df_general_1 = df_general_1[df_general_1[TAG] == PHASE]
    df_general_2 = df_general_2[df_general_2[TAG] == PHASE]

    """ GENERAL 2 DATA HANDLING """
    # We add a new column "Duration of treatment" (DT) (years)
    AbsTimeLapse(df_general_2, DT, DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END)

    # We adjust the column "Radiotherapy dose" because it is null if "Radiotherapy?" column equals 0
    df_general_2.loc[df_general_2[RADIOTHERAPY] == '0.0', RADIOTHERAPY_DOSE] = 0

    # We delete unnecessary variables from the dataframe
    df_general_2 = df_general_2.drop([RADIOTHERAPY, DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END], axis=1)

    # We reorder columns to have complications at the end
    cols = list(df_general_2.columns.values)
    cols = cols[-1:]+cols[:-1]
    df_general_2 = df_general_2[cols]

    """ DEX_DOX DATA HANDLING """
    # We add a new categorical column based on DEX cumulative dose
    conditions = [(~DEX_DOX[DEX].isnull()), (DEX_DOX[DEX].isnull())]
    categories = ["Yes", "Unknown"]

    DEX_DOX[DEX_PRESENCE] = select(conditions, categories)

    # We delete unnecessary variables from the dataframe
    DEX_DOX = DEX_DOX.drop([DEX], axis=1)

    """ DATAFRAME CONCATENATION """
    # We concatenate all the dataframes
    complete_df = pd.merge(DEX_DOX, df_general_1, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_general_2, on=PKEY, how=INNER)

    # We remove TAG variable
    complete_df = complete_df.drop([TAG], axis=1)

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)  # 9 values out of 13x251 = 3263 (0.2%)

    """ TABLE CREATION """
    types = {c: TYPES[c] for c in complete_df.columns}

    # We create the table
    data_manager.create_and_fill_table(complete_df, FIXED_FEATURES_AND_COMPLICATIONS,
                                       types=types, primary_key=[PARTICIPANT])
