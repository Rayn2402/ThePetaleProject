"""
Filename: replicate_study.py

Author: Nicolas Raymond

Description: Script used to replicate the entire study

Date of last modification: 2022/07/14
"""

from argparse import ArgumentParser
from os.path import join
from settings.paths import Paths
from src.data.extraction.constants import TDM6_DIST, TDM6_HR_END
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.sampling import get_obesity_data, get_VO2_data
from src.utils.argparsers import print_arguments
from subprocess import check_call

# TASK CHOICES
VO2: str = 'vo2'
OB: str = 'obesity'

# DATA SOURCE CHOICES
ORIGINAL: str = 'original'
GENERATED: str = 'generated'


def argument_parser():
    """
    Creates a parser for the replication of the study
    """

    # Parser creation
    parser = ArgumentParser(usage='python replicate_study.py -task ["vo2" or "obesity"]'
                                  ' -data ["original" or "generated]')

    # Definition of arguments
    parser.add_argument('-d', '--data', type=str, default=ORIGINAL, choices=[ORIGINAL, GENERATED],
                        help=f'Choice of the source of data (default = {ORIGINAL})')
    parser.add_argument('-t', '--task', type=str, default=VO2, choices=[VO2, OB],
                        help=f'Choice of prediction task (default = {VO2})')

    # Argument parsing
    arguments = parser.parse_args()
    print_arguments(arguments)

    return arguments


if __name__ == '__main__':

    args = argument_parser()

    """
    1. Data loading
    """
    # If we want to use the same data as in the current study
    if args.data == ORIGINAL:

        # Initialization of the data manager
        data_manager = PetaleDataManager()

        # Table extraction
        if args.task == VO2:
            df, target, cont_cols, cat_cols = get_VO2_data(data_manager, baselines=True, sex=True)
            df.drop([TDM6_DIST, TDM6_HR_END], axis=1, inplace=True)
            cont_cols.remove(TDM6_DIST)
            cont_cols.remove(TDM6_HR_END)

        else:
            df, target, cont_cols, cat_cols = get_obesity_data(data_manager, baselines=True, genomics=True)

    else:
        data = ""

    """
    2. Evaluation of the models
    """
    # Model arguments
    model_args = ['-rf', '-xg', '-enet', '-mlp', 'gcn', 'gat']

    # Graph arguments
    graph_degrees = ['-deg'] + [str(i) for i in range(5, 11)]
    graph_args = ['-cond_col'] + graph_degrees

    """
    -- 2.1 Evaluation with manually selected hyperparameters
    """
    # Evaluation
    script_path = join(Paths.EXPERIMENTS_SCRIPTS, args.task, 'manual_evaluations.py')
    experiment_args = ['-b', '-s', '-r_w', '-rf', '-xg', '-enet', '-mlp', 'gcn', 'gat']
    check_call(args=['python', script_path, *experiment_args])

    # Results compilation

    """
    -- 2.2 Evaluation with automated hyperparameter optimization
    """
    # Evaluation

    # Results compilation

    """
    3. Model analyses
    """
    # Loading of results

    # Selection of model

    """
    4. Final test on the holdout set
    """
    """
    -- 4.1 Test with manually selected hyperparameters
    """
    # Test

    # Results compilation
    """
    -- 4.2 Test with automated hyperparameter optimization
    """
    # Test

    # Results compilation
