"""
Filename: SNPS.py

Authors: Nicolas Raymond

Description: This file stores the procedure to create the
             tables containing the genomic data related to SNPs

Date of last modification : 2021/11/05
"""
import sys

from os.path import join, dirname, realpath
from pandas import read_csv
from typing import Any


def build_snp_table(directory: str,
                    table_name: str,
                    data_manager: Any) -> None:
    """
    Creates an snps table using original csv data

    Args:
        directory: directory containing the csv files
        table_name: name of the csv file containing the table (will be also used for table name)
        data_manager: PetaleDataManager

    Returns:

    """
    # We build the pandas dataframe
    df = read_csv(join(directory, f"{table_name}.csv"))

    # We change column names to have real patient IDs
    conversion_dict = data_manager.get_id_conversion_map()
    df.columns = df.columns.to_series().apply(lambda x: x.split("_")[0])
    df.columns = df.columns.to_series().apply(lambda x: conversion_dict.get(x, x))

    # We create the dictionary with column types
    types = {k: TYPES.get(k, CATEGORICAL_TYPE) for k in list(df.columns.values)}

    # We create and fill the table in the database
    data_manager.create_and_fill_table(df, table_name, types)


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import SNPS_COMMON, SNPS_RARE, CATEGORICAL_TYPE, TYPES
    from src.data.extraction.data_management import PetaleDataManager

    # We create a data manager
    dm = PetaleDataManager()

    # We build SNPS_RARE and SNPS_COMMON tables
    for t in [SNPS_RARE, SNPS_COMMON]:
        build_snp_table(Paths.CSV_FILES, t, dm)
