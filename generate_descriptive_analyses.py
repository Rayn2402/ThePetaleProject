"""
Filename: generate_descriptive_analyses.py

Author: Nicolas Raymond
        Mehdi Mitiche

Description: This file stores the procedures that needs to be executed in order to extract descriptive tables
             with information from all variables of a table.

Date of last modification: 2022/04/26
"""
from src.data.extraction.data_management import PetaleDataManager

if __name__ == '__main__':

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager()

    # We retrieve the name of the table concerned
    TABLE = input("Write the table name : ")

    # We proceed to the descriptive analysis of the variables of the table
    stats = data_manager.get_table_stats(TABLE)
    print(stats, "\n\n")

    # We count the missing data
    df, summary_dict = data_manager.get_missing_data_count(TABLE)
    print(df, "\n\n")
    print(summary_dict)
