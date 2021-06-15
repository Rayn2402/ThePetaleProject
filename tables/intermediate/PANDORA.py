
"""
Author : Nicolas Raymond

This file contains the procedure to create the table PANDORA containing the IDs conversion map
to match patient with genomic results.
"""

from os.path import join, realpath, dirname
from pandas import read_csv

import sys

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import PARTICIPANT, REF_NAME, TYPES, PETALE_PANDORA
    from src.data.extraction.data_management import initialize_petale_data_manager

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    df = read_csv(join(Paths.CSV_FILES.value, f"{PETALE_PANDORA}.csv"))

    # We modify data in the "Reference name" column to only keep the part after the last "-"
    df[REF_NAME] = df[REF_NAME].apply(lambda x: x.split("-")[-1].lstrip('0'))

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, PETALE_PANDORA,
                                       types={PARTICIPANT: TYPES[PARTICIPANT], REF_NAME: TYPES[REF_NAME]},
                                       primary_key=[PARTICIPANT])
