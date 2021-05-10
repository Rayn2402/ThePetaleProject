"""
Author : Nicolas Raymond

This file contains the procedure to create the tables containing the TOP12 SNPs that are
the most significantly associated to cardiorespiratory fitness.

"""

from SQL.DataManagement.Utils import initialize_petale_data_manager
from pandas import concat
from SQL.NewTablesScripts.constants import SIGNIFICANT_COMMON_SNPS_ID, GENES, SNPS_ID, SNPS_TYPE, \
    SNPS_POSITION, CHROM, TYPES, SIGNIFICANT_RARE_SNPS_ID, TOP12_SNPS_ID


COL = {GENES: TYPES[GENES], SNPS_TYPE: TYPES[SNPS_TYPE],
       SNPS_ID: TYPES[SNPS_ID], CHROM: TYPES[CHROM],
       SNPS_POSITION: TYPES[SNPS_POSITION]}
DIR = "../csv_files"
EXT = "csv"
SEP = ","


if __name__ == '__main__':

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
    dm.create_and_fill_table(top_12_df, TOP12_SNPS_ID, COL)