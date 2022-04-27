"""
Filename: get_model_predictions_csv.py

Author: Nicolas Raymond

Description: Script used to retrieved predictions from multiple records file

Date of last modification: 2022/04/27
"""

from argparse import ArgumentParser
from src.utils.argparsers import print_arguments
from src.utils.results_analysis import extract_predictions


def paths_and_ids_parser():
    """
    Provides an argparser that retrieves multiple paths, ids identifying them
    and a filename in which the predictions will be stored
    """
    # Create a parser
    parser = ArgumentParser(usage='\n python3 file.py -p [path1 path2 ... ] -ids [id1 id2 ... ]',
                            description="Stores multiple path")

    parser.add_argument('-p', '--paths', nargs='*', type=str, help='List of paths')
    parser.add_argument('-ids', '--ids', nargs='*', type=str, help='List of ids associated to the paths')
    parser.add_argument('-fn', '--filename', type=str, default='predictions',
                        help='Name of the file in which the predictions will be stored')
    arguments = parser.parse_args()
    print_arguments(arguments)

    return arguments


if __name__ == '__main__':

    # We retrieve paths and their ids
    args = paths_and_ids_parser()

    # We retrieve the predictions
    extract_predictions(paths=args.paths, model_ids=args.ids, filename=args.filename)
