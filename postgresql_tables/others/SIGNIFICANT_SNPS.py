"""
Filename: SIGNIFICANT_SNPS.py

Authors: Nicolas Raymond

Description: This file stores the procedure to create the tables
             containing details on SNPs significantly associated to cardiorespiratory fitness.

Date of last modification : 2021/11/05
"""
import sys

from os.path import realpath, dirname, join
from pandas import read_csv

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager

    # We create a data manager
    dm = PetaleDataManager()

    for t in [SIGNIFICANT_COMMON_SNPS_ID, SIGNIFICANT_RARE_SNPS_ID]:

        # We build the pandas dataframe
        df = read_csv(join(Paths.CSV_FILES, f"{t}.csv"))

        # We remove duplicate rows
        df = df.drop_duplicates()

        # We create and fill the table in the database
        dm.create_and_fill_table(df, t,
                                 types={GENES: TYPES[GENES],
                                        SNPS_ID: TYPES[SNPS_ID],
                                        CHROM: TYPES[CHROM],
                                        SNPS_POSITION: TYPES[SNPS_POSITION]})



