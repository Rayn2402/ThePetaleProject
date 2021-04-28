"""
Author : Nicolas Raymond

This file contains the procedure to create the tables containing the genomic data related to SNPs.

"""

from SQL.DataManager.Utils import initialize_petale_data_manager
from pandas import read_csv
from constants import SNPS_COMMON, SNPS_RARE
import os

DIR = "csv_files"
EXT = "csv"
SEP = ","


def build_snp_tables(table_name, data_manager):
    """
    Creates SNPs table using original csv data
    :param table_name: name of the csv file containing the table (will be also used for table name)
    :param data_manager: PetaleDataManager
    :return:
    """

    # We build the pandas dataframe
    df = read_csv(os.path.join(DIR, f"{table_name}.{EXT}"), sep=SEP)

    # We change column names to have real patient IDs
    conversion_dict = data_manager.get_id_conversion_map()
    df.columns = df.columns.to_series().apply(lambda x: x.split("_")[0])
    df.columns = df.columns.to_series().apply(lambda x: conversion_dict.get(x, x))

    # We create the dictionary with column types
    COL = {k: "text" for k in list(df.columns.values)}

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, table_name, COL)


if __name__ == '__main__':

    # We create a data manager
    dm = initialize_petale_data_manager()

    # We build SNPS_RARE and SNPS_COMMON tables
    for t in [SNPS_RARE, SNPS_COMMON]:
        build_snp_tables(t, dm)
