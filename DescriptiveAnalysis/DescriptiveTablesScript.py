"""
Authors : Nicolas Raymond
          Mehdi Mitiche

This file store the procedures that need to be executed in order to get descriptive tables
with information from all variables considered for our models (variables from Learning_1).
"""
from SQLutils.Utils import PetaleDataManager

TABLE = "Learning_1_GeneralsAndCardio0"

# We build a PetaleDataManager that will help interacting with PETALE database
user_name = input("Enter your username to access PETALE database : ")
data_manager = PetaleDataManager(user_name)

# We proceed to the descriptive analysis of the variables of the table
stats = data_manager.get_table_stats(TABLE)
print(stats, "\n\n")

# We count the missing data
missing = data_manager.get_missing_data_count(TABLE, drawChart=True)
print(missing)
