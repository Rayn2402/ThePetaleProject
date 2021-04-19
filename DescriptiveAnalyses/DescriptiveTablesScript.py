"""
Authors : Nicolas Raymond
          Mehdi Mitiche

This file store the procedures that needs to be executed in order to extract descriptive tables
with information from all variables of a table.
"""

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from SQL.DataManager.Utils import PetaleDataManager


# We build a PetaleDataManager that will help interacting with PETALE database
user_name = input("Enter your username to access PETALE database : ")
data_manager = PetaleDataManager(user_name)

# We retrieve the name of the table concerned
TABLE = input("Write table name : ")

# We proceed to the descriptive analysis of the variables of the table
stats = data_manager.get_table_stats(TABLE)
print(stats, "\n\n")

# We count the missing data
missing = data_manager.get_missing_data_count(TABLE, drawChart=True)
print(missing)
