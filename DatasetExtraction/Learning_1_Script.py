"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain "Learning_1 General" table in the database.
This table will consist of one of the simplest dataset that we will use to train our different model.
"""

from SQLutils import PetaleDataManager
from SQLutils import AbsTimeLapse
import pandas as pd


# We build a PetaleDataManager that will help interacting with PETALE database
user_name = input("Enter your username to access PETALE database : ")
data_manager = PetaleDataManager(user_name)

# We save the variables from General_2 with which we will create a table
General_2_vars = {"Date": "date",
                  "Participant": "text",
                  "Tag": "text",
                  "34472 Age at diagnosis": "numeric",
                  "34471 Date of diagnosis": "date",
                  "34474 Date of treatment end": "date",
                  "34475 Risk group": "text",
                  "34477 Boston protocol followed": "text",
                  "34479 Radiotherapy?": "text",
                  "34480 Radiotherapy dose": "numeric"}

# We save the variables from General_1 with which we will create a table
General_1_vars = {"Date": "date",
                  "Participant": "text",
                  "Tag": "text",
                  "34500 Sex": "text",
                  "34502 Height": "numeric",
                  "34503 Weight": "numeric",
                  "34604 Is currently smoking?": "text"}

# We save the variables from Cardio_0 with which we will create a table
# numerical_vars_name = data_manager.get_column_names("Cardio_0_Évaluation à l'Effort (EE)")
# numerical_vars_name = [var for var in numerical_vars_name if var not in ['Date', 'Participant', 'Status',
#                                                                          '35006 EE_VO2_max', 'Form', 'Day of Study',
#                                                                          'Tag', 'Remarks']]
Cardio_0_vars = {"Date": "date", "Participant": "text", "Tag": "text"}

# for var in numerical_vars_name:
#     Cardio_0_vars[var] = "numeric"

Cardio_0_vars['35006 EE_VO2_max'] = "numeric"

# We get a dataframe from the each table concerned
df_general_1 = data_manager.get_table("General_1_Demographic Questionnaire", columns=General_1_vars.keys())
df_general_2 = data_manager.get_table("General_2_CRF Hematology-Oncology", columns=General_2_vars.keys())
df_cardio_0 = data_manager.get_table("Cardio_0_Évaluation à l'Effort (EE)", columns=Cardio_0_vars.keys())

# We only keep survivors from Phase 1
df_general_1 = df_general_1[df_general_1["Tag"] == "Phase 1"]
df_general_2 = df_general_2[df_general_2["Tag"] == "Phase 1"]
df_cardio_0 = df_cardio_0[df_cardio_0["Tag"] == "Phase 1"]

# We add a new column Time of treatment (months) to the table general_2
AbsTimeLapse(df_general_2, "Time of treatment", "34471 Date of diagnosis", "34474 Date of treatment end")

# We remove the column "Date of diagnosis"
df_general_2 = df_general_2.drop(["34471 Date of diagnosis"], axis=1)

# We rename the date column for each table
df_general_1 = df_general_1.rename(columns={"Date": "Date G1"})
df_general_2 = df_general_2.rename(columns={"Date": "Date G2"})
df_cardio_0 = df_cardio_0.rename(columns={"Date": "Date C0"})

# We concatenate the three dataframes
pkey = ["Participant", "Tag"]
complete_df = pd.merge(df_general_1, df_general_2, on=pkey, how="inner")
complete_df = pd.merge(complete_df, df_cardio_0, on=pkey, how="inner")

# We create new time variables using Date C0 as present time
AbsTimeLapse(complete_df, "Time between G2 and G1", "Date G1", "Date G2")
AbsTimeLapse(complete_df, "Time between C0 and G1", "Date G1", "Date C0")
AbsTimeLapse(complete_df, "Time since end of treatment", "34474 Date of treatment end", "Date C0")

# We remove date variables
complete_df = complete_df.drop(["Date C0", "Date G1", "Date G2", "34474 Date of treatment end"], axis=1)

# We look at the date differences between the tables
print("\nTime differences :")
print("abs(G2 - G1) : ", complete_df["Time between G2 and G1"].unique())
print("abs(C0 - G1) : ", complete_df["Time between C0 and G1"].unique())
print("\n")

# Since there are no differences, we remove both columns and consider Height, Weight and Smoking variables up to date
complete_df = complete_df.drop(["Time between G2 and G1", "Time between C0 and G1"], axis=1)

# We adjust the column "Radiotherapy dose" because it is null if "Radiotherapy?" column equals 0
complete_df.loc[complete_df['34479 Radiotherapy?'] == '0.0', '34480 Radiotherapy dose'] = 0

# We remove "Radiotherapy?" column since it is no longer needed
complete_df = complete_df.drop(["34479 Radiotherapy?"], axis=1)

# We concatenate dictionaries of variables and delete non needed ones
vars = dict(General_2_vars, **General_1_vars)
vars["Time of treatment"] = "numeric"
vars["Time since end of treatment"] = "numeric"
vars = dict(vars, **Cardio_0_vars)
vars.pop("Date")
vars.pop("34471 Date of diagnosis")
vars.pop("34474 Date of treatment end")
vars.pop("34479 Radiotherapy?")

# We filter the dataframe created
complete_df = complete_df[vars.keys()]

# We create the table
data_manager.create_and_fill_table(complete_df, "Learning_1 General Data", types=vars, primary_key=pkey)
