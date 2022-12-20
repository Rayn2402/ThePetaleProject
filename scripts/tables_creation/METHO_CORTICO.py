"""
Filename: METHO_CORTICO.py

Author: Nicolas Raymond

Description: Stores the procedure to create the table with methotrexate
             and corticosteroids doses.

Date of last modification: 2022/03/23
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
    df = read_csv(join(Paths.CSV_FILES, f"{METHO_CORTICO_TABLE}.csv"), sep=";")
    df[PARTICIPANT] = df[PARTICIPANT].astype('string').apply(data_manager.fill_participant_id)

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, METHO_CORTICO_TABLE,
                                       types={PARTICIPANT: TYPES[PARTICIPANT],
                                              METHO: TYPES[METHO],
                                              CORTICO: TYPES[CORTICO]})

