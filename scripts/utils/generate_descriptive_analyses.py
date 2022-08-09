"""
Filename: generate_descriptive_analyses.py

Author: Nicolas Raymond
        Mehdi Mitiche

Description: Script that conducts a descriptive analysis of a table

Date of last modification: 2022/07/12
"""
import sys
from os.path import dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from src.data.extraction.data_management import PetaleDataManager

    # We build a data manager
    data_manager = PetaleDataManager()

    # We retrieve the name of the table concerned
    TABLE = input("Write the table name : ")

    # We proceed to the descriptive analysis of of the table
    stats = data_manager.get_table_stats(TABLE)
    print(stats, "\n\n")

    # We count the missing data
    df, summary_dict = data_manager.get_missing_data_count(TABLE)
    print(df, "\n\n")
    print(summary_dict)
