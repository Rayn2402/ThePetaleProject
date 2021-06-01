"""
Author : Nicolas Raymond

This file stores the procedure to execute in order to obtain "BASE_FEATURES_AND_COMPLICATIONS"
table in the database.

 FIXED_FEATURES_AND_COMPLICATIONS contains :

    Features:
     - SEX
     - AGE AT DIAGNOSIS
     - DURATION OF TREATMENT (DT)
     - RADIOTHERAPY DOSE (0; >0)
     - DOX DOSE
     - DEX (0; >0, <=Med; >Med) where Med is the median without 0's
     - GESTATIONAL AGE AT BIRTH (<37w, >=37w, NaN)
     - WEIGHT AT BIRTH (<2500g, >=2500g, NaN)

    Complications:
    - Neurocognitive (0: No, 1: Yes)
    - Skeletal/Bone (0: No, 1: Yes)
    - Cardiometabolic (0: No, 1: Yes)


"""

from SQL.DataManagement.Utils import initialize_petale_data_manager
from SQL.DataManagement.Helpers import AbsTimeLapse, get_missing_update
from SQL.constants import *
from numpy import select
from numpy import minimum as npmin
from numpy import nan_to_num
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

    # We retrieve the tables with baseline features and the table with invalid IDS
    df_general_1 = data_manager.get_table(GEN_1, G1_vars)
    df_general_2 = data_manager.get_table(GEN_2, G2_vars)
    DEX_DOX = data_manager.get_table(DEX_DOX_TABLE, DEX_DOX_vars)
    invalid_ids = data_manager.get_table(INVALID_ID_TABLE)

    # We only keep survivors from Phase 1
    df_general_1 = df_general_1[df_general_1[TAG] == PHASE]
    df_general_2 = df_general_2[df_general_2[TAG] == PHASE]

    """ GENERAL 1 DATA HANDLING """
    # We change values of SEX according to their respective categories
    df_general_1[SEX] = df_general_1[SEX].apply(lambda a: SEX_CATEGORIES[int(float(a))])

    # We change values of BIRTH AGE according to their respective categories
    df_general_1[BIRTH_AGE] = df_general_1[BIRTH_AGE].apply(lambda ba: BIRTH_AGE_CATEGORIES[int(float(ba))])

    # We change values of BIRTH WEIGHT according to their respective categories
    df_general_1[BIRTH_WEIGHT] = df_general_1[BIRTH_WEIGHT].apply(lambda bw: BIRTH_WEIGHT_CATEGORIES[int(float(bw))])

    """ GENERAL 2 DATA HANDLING """
    # We combine metabolic and cardio complications
    cardio_metabolic = (nan_to_num(df_general_2[METABOLIC_COMPLICATIONS].astype(float).to_numpy()) +
                        nan_to_num(df_general_2[CARDIAC_COMPLICATIONS].astype(float).to_numpy()))

    df_general_2[CARDIOMETABOLIC_COMPLICATIONS] = npmin(cardio_metabolic, 1)

    # We combine all complications to create a single overall complication
    complications = (cardio_metabolic +
                     nan_to_num(df_general_2[BONE_COMPLICATIONS].astype(float).to_numpy()) +
                     nan_to_num(df_general_2[NEUROCOGNITIVE_COMPLICATIONS].astype(float).to_numpy()))

    df_general_2[COMPLICATIONS] = npmin(complications, 1)

    # We add a new column "Duration of treatment" (DT) (years)
    AbsTimeLapse(df_general_2, DT, DATE_OF_DIAGNOSIS, DATE_OF_TREATMENT_END)

    # We adjust the column "Radiotherapy dose" because it is null if "Radiotherapy?" column equals 0
    df_general_2.loc[df_general_2[RADIOTHERAPY] == '0.0', RADIOTHERAPY_DOSE] = "0"

    # We change "Radiotherapy dose" to be two categories "0" or ">0"
    df_general_2.loc[df_general_2[RADIOTHERAPY_DOSE] != "0", RADIOTHERAPY_DOSE] = ">0"

    # We delete unnecessary variables from the dataframe
    df_general_2 = df_general_2.drop([RADIOTHERAPY, DATE_OF_DIAGNOSIS,
                                      DATE_OF_TREATMENT_END, METABOLIC_COMPLICATIONS,
                                      CARDIAC_COMPLICATIONS], axis=1)

    # We reorder columns to have complications at the end
    cols = list(df_general_2.columns.values)
    cols = cols[-1:]+cols[:-1]
    df_general_2 = df_general_2[cols]

    """ DEX_DOX DATA HANDLING """
    # We create a categorical column based on DEX cumulative dose
    temp = DEX_DOX[~(DEX_DOX[DEX].isnull())]  # Temp dataframe with non null values
    median = temp[DEX].median()               # DEX median

    conditions = [(DEX_DOX[DEX] > median),
                  (DEX_DOX[DEX] <= median),
                  (DEX_DOX[DEX].isnull())]

    categories = [f">{median}", f">0, <={median}", "0"]
    DEX_DOX.loc[:, DEX] = select(conditions, categories)

    # We change null DOX values for 0
    DEX_DOX.loc[DEX_DOX[DOX].isnull(), DOX] = 0

    """ DATAFRAME CONCATENATION """
    # We concatenate all the dataframes
    complete_df = pd.merge(DEX_DOX, df_general_1, on=[PARTICIPANT], how=INNER)
    complete_df = pd.merge(complete_df, df_general_2, on=PKEY, how=INNER)

    # We remove TAG variable
    complete_df = complete_df.drop([TAG], axis=1)

    """ REMOVAL OF INVALID ID """
    complete_df = complete_df[~(complete_df[PARTICIPANT].isin(list(invalid_ids[PARTICIPANT].values)))]

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)  # 32 values out of 12x247 = 2964 (1%)

    """ TABLE CREATION """
    types = {c: TYPES[c] for c in complete_df.columns}

    # We create the table
    data_manager.create_and_fill_table(complete_df, BASE_FEATURES_AND_COMPLICATIONS,
                                       types=types, primary_key=[PARTICIPANT])
