"""
Filename: descriptive_analyses.py

Authors: Nicolas Raymond

Description: This file is used to generate descriptive analyses of the table
             related to the VO2 peak prediction task.

Date of last modification : 2022/07/07
"""

from src.data.extraction.constants import *
from src.data.extraction.data_management import PetaleDataManager

if __name__ == '__main__':

    # Data manager initialization
    data_manager = PetaleDataManager()

    # Tables extraction
    variables = [PARTICIPANT, SEX, TDM6_DIST, TDM6_HR_END, WEIGHT, DT, AGE, MVLPA, VO2R_MAX]
    learning_set = data_manager.get_table(VO2_LEARNING_SET, columns=variables)
    holdout_set = data_manager.get_table(VO2_HOLDOUT_SET, columns=variables)
    dataset = learning_set.append(holdout_set)

    # We proceed to the descriptive analyses
    data_manager.get_table_stats(learning_set, filename='vo2_learning_set')
    data_manager.get_table_stats(holdout_set, filename='vo2_holdout_set')
    data_manager.get_table_stats(dataset, filename='vo2_dataset')
