"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain "Learning_0_6MWT_and_Generals (WarmUp)" table in the database.
This table will consist of one of the dataset two reproduce 6MWT experiment with a more complex model.
"""

from SQL.DataManager.Utils import PetaleDataManager
from SQL.DataManager.Helpers import AbsTimeLapse
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
    GEN_1 = "General_1_Demographic Questionnaire"
    General_1_vars = {"Date": "date",
                      "Participant": "text",
                      "Tag": "text",
                      "34501 Date of birth (survivor)": "date",
                      "34503 Weight": "numeric"}

    # We save the variables needed from General_2
    GEN_2 = "General_2_CRF Hematology-Oncology"
    General_2_vars = {"Participant": "text",
                      "Tag": "text",
                      "34471 Date of diagnosis": "date",
                      "34474 Date of treatment end": "date"}

    # We save the variables needed from Cardio_0
    CARDIO_0 = "Cardio_0_Évaluation à l'Effort (EE)"
    Cardio_0_vars = {"Participant": "text",
                     "Tag": "text",
                     "35009 EE_VO2r_max": "numeric"}

    # We save the variables needed from Cardio_3
    CARDIO_3 = "Cardio_3_Questionnaire d'Activité Physique (QAP)"
    Cardio_3_vars = {"Participant": "text",
                     "Tag": "text",
                     "35116 QAPL8": "numeric"}

    # We save the variables needed from Cardio_4
    CARDIO_4 = "Cardio_4_Test de Marche de 6 Minutes (TDM6)"
    Cardio_4_vars = {"Participant": "text",
                     "Tag": "text",
                     "35149 TDM6_HR_6_2": "numeric",
                     "35142 TDM6_Distance_2": "numeric"}

    # We save some helpful constants
    KEY1, KEY2 = "Participant", "Tag"
    TAG, PHASE = "Tag", "Phase 1"
    DIAGNOSIS = "34471 Date of diagnosis"
    TREATMENT_END = "34474 Date of treatment end"
    DT = "Duration of treatment"
    MVLPA, QAPL8 = "MVLPA", "35116 QAPL8"
    INNER = "inner"
    PRESENT, BIRTH, AGE = "Date", "34501 Date of birth (survivor)", "Age"
    VO2_MAX = '35009 EE_VO2r_max'
    ID_TABLE = "VO2_ID"

    # We retrieve the tables with variables and the table with the filtered ID
    df_general_1 = data_manager.get_table(GEN_1, General_1_vars.keys())
    df_general_2 = data_manager.get_table(GEN_2, General_2_vars.keys())
    df_cardio_0 = data_manager.get_table(CARDIO_0, Cardio_0_vars.keys())
    df_cardio_3 = data_manager.get_table(CARDIO_3, Cardio_3_vars.keys())
    df_cardio_4 = data_manager.get_table(CARDIO_4, Cardio_4_vars.keys())
    IDs = data_manager.get_table(ID_TABLE)

    # We only keep survivors from Phase 1
    df_general_1 = df_general_1[df_general_1[TAG] == PHASE]
    df_general_2 = df_general_2[df_general_2[TAG] == PHASE]
    df_cardio_0 = df_cardio_0[df_cardio_0[TAG] == PHASE]
    df_cardio_3 = df_cardio_3[df_cardio_3[TAG] == PHASE]
    df_cardio_4 = df_cardio_4[df_cardio_4[TAG] == PHASE]

    # We remove survivors that have missing VO2r_max value
    df_cardio_0 = df_cardio_0[~df_cardio_0[VO2_MAX].isnull()]

    # We add a new column Duration of treatment (years) to the table general_2
    AbsTimeLapse(df_general_2, DT, DIAGNOSIS, TREATMENT_END)

    # We sum the two column QALP3 and QALP4 to get the sum of minutes
    df_cardio_3 = df_cardio_3.rename(columns={QAPL8: MVLPA})

    # We concatenate all the dataframes
    pkey = [KEY1, KEY2]
    complete_df = pd.merge(IDs, df_general_1)
    complete_df = pd.merge(complete_df, df_general_2, on=pkey, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_0, on=pkey, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_3, on=pkey, how=INNER)
    complete_df = pd.merge(complete_df, df_cardio_4, on=pkey, how=INNER)

    # We add the "Age" column
    AbsTimeLapse(complete_df, AGE, BIRTH, PRESENT)

    # We remove the dates column and the "Tag" column
    complete_df = complete_df.drop([PRESENT, TREATMENT_END, DIAGNOSIS, BIRTH, KEY2], axis=1)

    # We concatenate dictionaries of variables and delete non needed ones
    vars = {**General_2_vars, **General_1_vars, **Cardio_3_vars, **Cardio_4_vars}

    for new_col in [DT, AGE, MVLPA]:
        vars[new_col] = "numeric"

    vars = {**vars, **Cardio_0_vars}

    for deleted_col in [PRESENT, QAPL8, TREATMENT_END, DIAGNOSIS, BIRTH, KEY2]:
        vars.pop(deleted_col)

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)  # 19 continuous values are missing

    # We filter the dataframe created
    complete_df = complete_df[vars.keys()]

    # We create the table
    data_manager.create_and_fill_table(complete_df, "Learning_0_6MWT_and_Generals (WarmUp)",
                                       types=vars, primary_key=[KEY1])

