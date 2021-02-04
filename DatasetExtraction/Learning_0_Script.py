"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain "Learning_0_6MWT_and_Generals (WarmUp)" table in the database.
This table will consist of one of the dataset two reproduce 6MWT experiment with a more complex model.
"""

from SQLutils import PetaleDataManager
from SQLutils import AbsTimeLapse
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
    General_2_vars = {"Date": "date",
                      "Participant": "text",
                      "Tag": "text",
                      "34471 Date of diagnosis": "date",
                      "34474 Date of treatment end": "date"}

    # We save the variables needed from Cardio_0
    CARDIO_0 = "Cardio_0_Évaluation à l'Effort (EE)"
    Cardio_0_vars = {"Date": "date",
                     "Participant": "text",
                     "Tag": "text",
                     "35009 EE_VO2r_max": "numeric"}

    # We save the variables needed from Cardio_3
    CARDIO_3 = "Cardio_3_Questionnaire d'Activité Physique (QAP)"
    Cardio_3_vars = {"Date": "date",
                     "Participant": "text",
                     "Tag": "text",
                     "35111 QAPL3": "numeric",
                     "35112 QAPL4": "numeric"}

    # We save the variables needed from Cardio_4
    CARDIO_4 = "Cardio_4_Test de Marche de 6 Minutes (TDM6)"
    Cardio_4_vars = {"Date": "date",
                     "Participant": "text",
                     "Tag": "text",
                     "35136 TDM6_HR_6_1": "numeric",
                     "35129 TDM6_Distance_1": "numeric"}

    # We retrieve the tables
    df_general_1 = data_manager.get_table(GEN_1, columns=General_1_vars.keys())
    df_general_2 = data_manager.get_table(GEN_2, columns=General_2_vars.keys())
    df_cardio_0 = data_manager.get_table(CARDIO_0, columns=Cardio_0_vars.keys())
    df_cardio_3 = data_manager.get_table(CARDIO_3, columns=Cardio_3_vars.keys())
    df_cardio_4 = data_manager.get_table(CARDIO_4, columns=Cardio_4_vars.keys())

    # We only keep survivors from Phase 1
    PHASE = "Phase 1"
    df_general_1 = df_general_1[df_general_1["Tag"] == PHASE]
    df_general_2 = df_general_2[df_general_2["Tag"] == PHASE]
    df_cardio_0 = df_cardio_0[df_cardio_0["Tag"] == PHASE]
    df_cardio_3 = df_cardio_3[df_cardio_3["Tag"] == PHASE]
    df_cardio_4 = df_cardio_4[df_cardio_4["Tag"] == PHASE]

    # We remove survivors that have missing VO2_max value
    df_cardio_0 = df_cardio_0[~df_cardio_0['35009 EE_VO2r_max'].isnull()]

    # We add a new column Duration of treatment (years) to the table general_2
    DIAGNOSIS = "34471 Date of diagnosis"
    TREATMENT_END = "34474 Date of treatment end"
    DT = "Duration of treatment"
    AbsTimeLapse(df_general_2, DT, DIAGNOSIS, TREATMENT_END)

    # We remove survivors that has missing value from QAPL3 and QAPL4 (minutes of moderate to vigorous activities)
    QAPL3, QAPL4 = "35111 QAPL3", "35112 QAPL4"
    filter = ~df_cardio_3[QAPL3].isnull() & ~df_cardio_3[QAPL4].isnull()
    df_cardio_3 = df_cardio_3[filter]

    # We sum the two column QALP3 and QALP4 to get the sum of minutes
    MVLPA = "MVLPA"
    df_cardio_3[MVLPA] = df_cardio_3[QAPL3] + df_cardio_3[QAPL4]
    df_cardio_3 = df_cardio_3.drop([QAPL3, QAPL4], axis=1)

    # We rename the date column for each table
    df_general_1 = df_general_1.rename(columns={"Date": "Date G1"})
    df_general_2 = df_general_2.rename(columns={"Date": "Date G2"})
    df_cardio_0 = df_cardio_0.rename(columns={"Date": "Date C0"})
    df_cardio_3 = df_cardio_3.rename(columns={"Date": "Date C3"})
    df_cardio_4 = df_cardio_4.rename(columns={"Date": "Date C4"})

    # We concatenate all the dataframes
    pkey = ["Participant", "Tag"]
    complete_df = pd.merge(df_general_1, df_general_2, on=pkey, how="inner")
    complete_df = pd.merge(complete_df, df_cardio_0, on=pkey, how="inner")
    complete_df = pd.merge(complete_df, df_cardio_3, on=pkey, how="inner")
    complete_df = pd.merge(complete_df, df_cardio_4, on=pkey, how="inner")

    # Considering the date of 6MWT as the present date, we look at date differences between tables
    PRESENT = "Date C4"
    dates = ["Date G1", "Date G2", "Date C0", "Date C3"]
    diff_dates = ["C4-G1", "C4-G2", "C4-C0", "C4-C3"]

    for new_col, old_date in zip(diff_dates, dates):
        AbsTimeLapse(complete_df, new_col, old_date, PRESENT)

    # We look at the date differences between the tables
    print("\nTime differences :")
    print("abs(C4 - G1) : ", complete_df["C4-G1"].unique())
    print("abs(C4 - G2) : ", complete_df["C4-G2"].unique())
    print("abs(C4 - C0) : ", complete_df["C4-C0"].unique())
    print("abs(C4 - C3) : ", complete_df["C4-C3"].unique())
    print("\n")

    # Since no time differences are observed, we remove the new columns
    complete_df = complete_df.drop(diff_dates, axis=1)

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)

    # We add the Age column
    BIRTH = "34501 Date of birth (survivor)"
    AGE = "Age"
    AbsTimeLapse(complete_df, AGE, BIRTH, PRESENT)

    # We also remove date variables
    complete_df = complete_df.drop(dates+[PRESENT, TREATMENT_END, DIAGNOSIS, BIRTH], axis=1)

    # We concatenate dictionaries of variables and delete non needed ones
    vars = {**General_2_vars, **General_1_vars, **Cardio_3_vars, **Cardio_4_vars}
    vars[DT] = "numeric"
    vars[AGE] = "numeric"
    vars[MVLPA] = "numeric"
    vars = {**vars, **Cardio_0_vars}
    vars.pop("Date")
    vars.pop(QAPL3)
    vars.pop(QAPL4)
    vars.pop(TREATMENT_END)
    vars.pop(DIAGNOSIS)
    vars.pop(BIRTH)

    # We look at the number of rows and the total of missing values per column
    get_missing_update(complete_df)  # Only 4 continuous values are missing!

    # We filter the dataframe created
    complete_df = complete_df[vars.keys()]

    # We create the table
    data_manager.create_and_fill_table(complete_df,
                                       "Learning_0_6MWT_and_Generals (WarmUp)", types=vars, primary_key=pkey)

