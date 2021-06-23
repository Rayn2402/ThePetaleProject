"""
Author : Nicolas Raymond

This file contains the procedure to create the table with IDs of
participant that satisfies criteria of maximal effort.

IDs were sent by Maxime Caru

Criterias can be found in :

"Maximal cardiopulmonary exercise testing in childhood acute
 lymphoblastic leukemia survivors exposed to chemotherapy"

"""

from os.path import join, dirname, realpath
from pandas import read_csv

import sys

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import VO2_ID_TABLE, PARTICIPANT, TYPES
    from src.data.extraction.data_management import initialize_petale_data_manager
    from src.data.extraction.helpers import fill_id

    # We build a PetaleDataManager that will help us interacting with the database
    data_manager = initialize_petale_data_manager()

    # We build the pandas dataframe
    IDs = read_csv(join(Paths.CSV_FILES, f"{VO2_ID_TABLE}.csv"))
    IDs[PARTICIPANT] = IDs[PARTICIPANT].astype('string').apply(fill_id)

    # We create and fill the table in the database
    data_manager.create_and_fill_table(IDs, VO2_ID_TABLE, {PARTICIPANT: TYPES[PARTICIPANT]}, primary_key=[PARTICIPANT])
