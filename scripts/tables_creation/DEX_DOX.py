"""
Filename: DEX_DOX.py

Authors: Nicolas Raymond

Description: This file stores the procedure to create
             the table associated to DEX and DOX cumulative doses.

Date of last modification : 2021/11/05
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
    df = read_csv(join(Paths.CSV_FILES, f"{DEX_DOX_TABLE}.csv"), sep=";")

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, DEX_DOX_TABLE,
                                       types={PARTICIPANT: TYPES[PARTICIPANT],
                                              DEX: TYPES[DEX], DOX: TYPES[DOX]})
