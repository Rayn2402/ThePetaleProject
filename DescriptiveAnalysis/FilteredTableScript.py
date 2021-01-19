"""
Authors : Nicolas Raymond

This file executes a procedure to create a filtered version of concatenated Generale 1 and 2 tables
"""

from SQLutils import PetaleDataManager
from SQLutils import timeDeltaToMonths
import pandas as pd

# We build a PetaleDataManager that will help interacting with PETALE database
user_name = input("Enter your username to access PETALE database : ")
data_manager = PetaleDataManager(user_name)

# We save the variables with which we will create a table
General_2_vars = {"Participant": "text",
                  "Tag": "text",
                  "34472 Age at diagnosis": "numeric",
                  "34471 Date of diagnosis": "date",
                  "34474 Date of treatment end": "date",
                  "34475 Risk group": "text",
                  "34477 Boston protocol followed": "text",
                  "34479 Radiotherapy?": "text",
                  "34480 Radiotherapy dose": "numeric"}

General_1_vars = {"Participant": "text",
                  "Tag": "text",
                  "34500 Sex": "text",
                  "34502 Height": "numeric",
                  "34503 Weight": "numeric",
                  "34604 Is currently smoking?": "text"}

# We get a dataframe from the each table concerned
df_general_1 = data_manager.get_table("General_1_Demographic Questionnaire", columns=General_1_vars)
df_general_2 = data_manager.get_table("General_2_CRF Hematology-Oncology", columns=General_2_vars)

# We only keep survivors from Phase 1
df_general_1 = df_general_1[df_general_1["Tag"] == "Phase 1"]
df_general_2 = df_general_2[df_general_2["Tag"] == "Phase 1"]

# We add a new column Time of treatment (months) to the table general_2
new = "Time of treatment"
df_general_2[new] = df_general_2["34474 Date of treatment end"]\
                                    - df_general_2["34471 Date of diagnosis"]
df_general_2[new] = df_general_2[new].apply(timeDeltaToMonths)

# We concatenate the two dataframes
pkey = ["Participant", "Tag"]
complete_df = pd.merge(df_general_1, df_general_2, on=pkey, how="inner")

# We concatenate dictionaries of variables and delete non needed ones
vars = dict(General_2_vars, **General_1_vars)
vars[new] = "numeric"
vars.pop("34471 Date of diagnosis")
vars.pop("34474 Date of treatment end")

# We filter the dataframe created
complete_df = complete_df[vars.keys()]

# We create the table
data_manager.create_and_fill_table(complete_df, "General_4_FilteredData", types=vars, primary_key=pkey)




