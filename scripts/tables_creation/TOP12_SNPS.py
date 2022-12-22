"""
Filename: TOP12_SNPS.py

Authors: Nicolas Raymond

Description: This file stores the procedure to create the tables
             containing the TOP12 SNPs that are the most significantly
             associated to cardiorespiratory fitness.

Date of last modification : 2021/11/05
"""
import sys

from os.path import dirname, realpath
from pandas import concat

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.extraction.constants import *

    # We create a data manager
    dm = PetaleDataManager()

    # We load SNPs tables from the database
    common_df = dm.get_table(SIGNIFICANT_COMMON_SNPS_ID)
    rare_df = dm.get_table(SIGNIFICANT_RARE_SNPS_ID)

    # We had column TYPE
    common_df[SNPS_TYPE] = ["COMMON"]*common_df.shape[0]
    rare_df[SNPS_TYPE] = ["RARE"]*rare_df.shape[0]

    # We stack both dataframes
    top_12_df = concat([common_df, rare_df])

    # We create and fill the table in the database
    dm.create_and_fill_table(top_12_df, TOP12_SNPS_ID,
                             types={GENES: TYPES[GENES], SNPS_TYPE: TYPES[SNPS_TYPE],
                                    SNPS_ID: TYPES[SNPS_ID], CHROM: TYPES[CHROM],
                                    SNPS_POSITION: TYPES[SNPS_POSITION]})
