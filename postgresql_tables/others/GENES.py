"""
Filename: GENES.py

Authors: Nicolas Raymond

Description: This file stores the procedure to create the pivoted tables containing
             the patients genomic data related to the top 5 and the top 12 SNPs most
             significantly associated to cardiorespiratory fitness.


Date of last modification : 2021/11/05
"""
import sys

from os.path import realpath, dirname
from pandas import merge

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.extraction.helpers import get_missing_update

    # We build a PetaleDataManager
    dm = PetaleDataManager()

    # We load SNPS_COMMON and SNPS_RARE tables
    common_df = dm.get_table(SNPS_COMMON)
    rare_df = dm.get_table(SNPS_RARE)

    for top_snps_table, new_table_name in [(TOP5_SNPS_ID, GENES_5), (TOP12_SNPS_ID, GENES_12), (None, ALLGENES)]:

        if top_snps_table is not None:

            # We load TOP_SNPS table and add a column concatenating CHROM and POS
            top_snps_df = dm.get_table(top_snps_table)
            top_snps_df[dm.KEY] = top_snps_df[CHROM].astype(str) + "_" + top_snps_df[SNPS_POSITION].astype(str)

            # We dump CHROM and POS columns
            top_snps_df = top_snps_df.drop([CHROM, SNPS_POSITION], axis=1)
            top_common_snps_df = top_snps_df[top_snps_df[SNPS_TYPE] == 'COMMON']
            top_rare_snps_df = top_snps_df[top_snps_df[SNPS_TYPE] == 'RARE']

        else:
            top_common_snps_df, top_rare_snps_df = None, None

        # For both dataframes we execute some preprocessing to enable
        common_df_copy = dm.pivot_snp_dataframe(common_df.copy(), top_common_snps_df)
        rare_df_copy = dm.pivot_snp_dataframe(rare_df.copy(), top_rare_snps_df)

        # We concat both dataframes
        gen_df = merge(common_df_copy, rare_df_copy, on=[PARTICIPANT], how=INNER, suffixes=('', '_y'))
        gen_df.drop(list(gen_df.filter(regex='_y$')), axis=1, inplace=True)
        get_missing_update(gen_df)

        # We create types dictionary to create the official table
        types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(gen_df.columns.values)}

        # We create and fill the table in the database
        dm.create_and_fill_table(gen_df, new_table_name, types, primary_key=[PARTICIPANT])
