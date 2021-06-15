"""
Author : Nicolas Raymond

This file contains the procedure to create the tables containing the TOP12 SNPs that are
the most significantly associated to cardiorespiratory fitness.

"""

from os.path import dirname, realpath
from pandas import concat

import sys

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from src.data.extraction.data_management import initialize_petale_data_manager
    from src.data.extraction.constants import *

    # We create a data manager
    dm = initialize_petale_data_manager()

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
