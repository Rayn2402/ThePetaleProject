"""
Filename: argparsers.py

Author: Nicolas Raymond

Description: This file stores common argparser functions

Date of last modification: 2022/08/02
"""

import argparse


def data_source_parser():
    """
    Creates a parser for the selection of a data source
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python [file].py',
                                     description="Enables to select data source")

    # Data source
    parser.add_argument('-from_csv', '--from_csv', default=False, action='store_true',
                        help='If true, extract the data from the csv file instead of the database.')

    return parser.parse_args()


def path_parser():
    """
    Provides an argparser that retrieves a path
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 file.py -p [path]',
                                     description="Stores a path")

    parser.add_argument('-p', '--path', type=str, help='Path of the experiment folder')

    return parser.parse_args()

