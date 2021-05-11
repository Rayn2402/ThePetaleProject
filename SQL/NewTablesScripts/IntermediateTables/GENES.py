"""
Author : Nicolas Raymond

This file contains the procedure to create the pivoted tables containing the patients genomic data
related to the top 5 and the top 12 SNPs most significantly associated to cardiorespiratory fitness.

"""

from SQL.DataManagement.Utils import initialize_petale_data_manager
from SQL.DataManagement.Helpers import get_missing_update
from SQL.constants import *
from pandas import merge

KEY = "CHROM_POS"


def pivot_snp_dataframe(df, snps_id_filter):
    """
    Filter the patients snps table and execute a pivot (transposition)

    :param df: pandas dataframe
    :param snps_id_filter: list with snps id to keep
    :return: pandas dataframe
    """
    df[KEY] = df[CHROM].astype(str) + "_" + df[SNPS_POSITION].astype(str)

    # We filter the table to only keep rows where CHROM and POS match with an SNP in the top 5
    df = df[df[KEY].isin(list(snps_id_filter[KEY].values))]

    # We dump CHROM and POS columns
    df = df.drop([CHROM, SNPS_POSITION, REF, ALT, GENE_REF_GEN], axis=1, errors='ignore')

    # We change index for CHROM_POS column
    df = df.set_index(KEY)

    # We transpose the dataframe
    df = df.T
    df.index.rename(PARTICIPANT, inplace=True)
    df.reset_index(inplace=True)

    return df


if __name__ == '__main__':

    # We build a PetaleDataManager
    dm = initialize_petale_data_manager()

    for top_snps_table, new_table_name in [(TOP5_SNPS_ID, GENES_5), (TOP12_SNPS_ID, GENES_12)]:

        # We load TOP5 table and add a column concatenating CHROM and POS
        top_snps_df = dm.get_table(top_snps_table)
        top_snps_df[KEY] = top_snps_df[CHROM].astype(str) + "_" + top_snps_df[SNPS_POSITION].astype(str)

        # We dump CHROM and POS columns
        top_snps_df = top_snps_df.drop([CHROM, SNPS_POSITION], axis=1)

        # We load SNPS_COMMON and SNPS_RARE tables
        common_df = dm.get_table(SNPS_COMMON)
        rare_df = dm.get_table(SNPS_RARE)

        # For both dataframes we execute some preprocessing to enable pivot
        common_df = pivot_snp_dataframe(common_df, top_snps_df)
        rare_df = pivot_snp_dataframe(rare_df, top_snps_df[top_snps_df[SNPS_TYPE] == 'RARE'])

        # We concat both dataframes
        gen_df = merge(common_df, rare_df, on=[PARTICIPANT], how=INNER)
        get_missing_update(gen_df)

        # We create types dictionary to create the official table
        types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(gen_df.columns.values)}

        # We create and fill the table in the database
        dm.create_and_fill_table(gen_df, new_table_name, types, primary_key=[PARTICIPANT])
