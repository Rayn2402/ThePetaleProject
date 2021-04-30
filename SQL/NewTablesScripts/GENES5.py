"""
Author : Nicolas Raymond

This file contains the procedure to create the pivoted tables containing the patients genomic data
related to the top 5 SNPs most significantly associated to cardiorespiratory fitness.

"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from SQL.NewTablesScripts.constants import SNPS_COMMON, SNPS_RARE, TOP5_SNPS_ID, CHROM,\
    SNPS_POSITION, REF, ALT, PARTICIPANT, GENE_REF_GEN, INNER, CATEGORICAL_TYPE, TYPES, GENES_5
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

    # We load TOP5 table and add a column concatenating CHROM and POS
    top_5_df = dm.get_table(TOP5_SNPS_ID)
    top_5_df[KEY] = top_5_df[CHROM].astype(str) + "_" + top_5_df[SNPS_POSITION].astype(str)

    # We dump CHROM and POS columns
    top_5_df = top_5_df.drop([CHROM, SNPS_POSITION], axis=1)

    # We load SNPS_COMMON and SNPS_RARE tables
    common_df = dm.get_table(SNPS_COMMON)
    rare_df = dm.get_table(SNPS_RARE)

    # For both dataframes we execute some preprocessing to enable pivot
    common_df = pivot_snp_dataframe(common_df, top_5_df)
    rare_df = pivot_snp_dataframe(rare_df, top_5_df)

    # We concat both dataframes
    gen_5_df = merge(common_df, rare_df, on=[PARTICIPANT], how=INNER)

    # We create types dictionary to create the official table
    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(gen_5_df.columns.values)}

    # We create and fill the table in the database
    dm.create_and_fill_table(gen_5_df, GENES_5, types, primary_key=[PARTICIPANT])