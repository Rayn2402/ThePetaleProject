"""
Filename: TOP5_SNPS.py

Authors: Nicolas Raymond

Description: This file stores the procedure to create the tables
             containing the TOP5 SNPs that are the most significantly
             associated to cardiorespiratory fitness.

Date of last modification : 2021/11/05
"""
import sys

from os.path import join, dirname, realpath
from pandas import read_csv, concat

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.extraction.constants import *

    # We create a data manager
    dm = PetaleDataManager()

    # We build the pandas dataframe containing the TOP5 most significantly associated SNPS
    top_snps_df = read_csv(join(Paths.CSV_FILES, f"{TOP5_SNPS_ID}.csv"))

    # We load SNPs tables from the database
    common_df = dm.get_table(SIGNIFICANT_COMMON_SNPS_ID)
    rare_df = dm.get_table(SIGNIFICANT_RARE_SNPS_ID)

    # We had column TYPE
    common_df[SNPS_TYPE] = ["COMMON"]*common_df.shape[0]
    rare_df[SNPS_TYPE] = ["RARE"]*rare_df.shape[0]

    # We filter the common and the rare SNPs
    common_df = common_df[common_df[SNPS_ID].isin(list(top_snps_df[SNPS_ID].values))]
    rare_df = rare_df[rare_df[SNPS_ID].isin(list(top_snps_df[SNPS_ID].values))]

    # We stack both dataframes
    top_5_df = concat([common_df, rare_df])

    # We create and fill the table in the database
    dm.create_and_fill_table(top_5_df, TOP5_SNPS_ID,
                             types={GENES: TYPES[GENES],
                                    SNPS_TYPE: TYPES[SNPS_TYPE],
                                    SNPS_ID: TYPES[SNPS_ID],
                                    CHROM: TYPES[CHROM],
                                    SNPS_POSITION: TYPES[SNPS_POSITION]})
