"""
Filename: EOT_BMI.py

Authors: Nicolas Raymond

Description: This file stores the procedure to create
             the table associated to the BMI at the end of treatment (EOT)

Date of last modification : 2022/06/06
"""
import sys

from os.path import join, dirname, realpath
from pandas import read_csv

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager()

    # We build the pandas dataframe
    df = read_csv(join(Paths.CSV_FILES, f"{EOT_BMI_TABLE}.csv"), sep=",")

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, EOT_BMI_TABLE,
                                       types={PARTICIPANT: TYPES[PARTICIPANT],
                                              EOT_BMI: TYPES[EOT_BMI]})
