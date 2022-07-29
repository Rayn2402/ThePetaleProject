"""
Filename: replicate_study.py

Author: Nicolas Raymond

Description: Script used to replicate the entire study

Date of last modification: 2022/07/28
"""

from argparse import ArgumentParser
from os import environ, listdir, makedirs, rename
from os.path import exists, join
from pandas import DataFrame, read_csv
from settings.paths import Paths
from shutil import move
from src.data.extraction.constants import DUMMY
from src.utils.metrics import AbsoluteError, Direction, ConcordanceIndex, Pearson,\
    RootMeanSquaredError, Sensitivity, Specificity
from subprocess import check_call
from typing import List, Tuple

environ['MKL_THREADING_LAYER'] = 'GNU'

# TASK CHOICES
VO2: str = 'vo2'
OB: str = 'obesity'

# DATA SOURCE CHOICES
ORIGINAL: str = 'original'
GENERATED: str = 'generated'

# GNN CHOICE
GAT = 'GAT'
GCN = 'GCN'


def add_delimiter(section: str) -> None:
    """
    Print a pattern to indicate a new section of the script

    Args:
        section: name of the new section

    Returns: None
    """
    print(f"\n{'*'*5} {section} {'*'*5}\n")


def argument_parser():
    """
    Creates a parser for the replication of the study
    """
    # Parser creation
    parser = ArgumentParser(usage='python replicate_study.py -task ["vo2" or "obesity"]'
                                  ' -data ["original" or "generated"]')

    # Definition of arguments
    parser.add_argument('-d', '--data', type=str, default=ORIGINAL, choices=[ORIGINAL, GENERATED],
                        help=f'Choice of the source of data (default = {ORIGINAL})')
    parser.add_argument('-t', '--task', type=str, default=VO2, choices=[VO2, OB],
                        help=f'Choice of prediction task (default = {VO2})')

    # Argument parsing
    arguments = parser.parse_args()

    return arguments


def reformat_scores_df(cell_content: str) -> float:
    """
    Functions that reformat cell of a pandas dataframe for further analyses

    Args:
        cell_content: str contained in a dataframe cell

    Returns: float
    """
    temp_list = cell_content.split(' +- ')
    mean = float(temp_list[0])
    std = float(temp_list[1])
    return mean + (std * 0.01)


def extract_best_graph_degree(df: DataFrame, model: str) -> int:
    """
    Finds the the degree of graph that allowed to achieve the best score

    Args:
        df: pandas dataframe with the scores of all models
        model: GCN or GAT

    Returns: degree (int)
    """
    # Validation of input model
    if model not in [GAT, GCN]:
        raise ValueError(f'Model must be in {[GAT, GCN]}')

    # We filter the dataframe to only consider the row associated to the model
    filtered_df = df.filter(like=model, axis=0)

    # We find the best model among the filtered dataframe
    best = find_best_model(filtered_df)

    # We return the degree associated to the model
    return int(best[3:])


if __name__ == '__main__':

    args = argument_parser()

    """
    0. Variables setup and definition of function specific to the task
    """
    # We set main variables according to the prediction task
    metrics = [AbsoluteError(), ConcordanceIndex(), Pearson(), RootMeanSquaredError()]
    model_args = ['-xg']

    if args.task == OB:
        learning_set_path = Paths.OBESITY_LEARNING_SET_CSV
        holdout_set_path = Paths.OBESITY_HOLDOUT_SET_CSV
        mask_path = Paths.OBESITY_MASK
        feature_args = ['-b', 'f']
    else:
        learning_set_path = Paths.VO2_LEARNING_SET_CSV
        holdout_set_path = Paths.VO2_HOLDOUT_SET_CSV
        mask_path = Paths.VO2_MASK
        feature_args = ['-b', '-s', '-r_w']
        #metrics += [Sensitivity(), Specificity()]

    metrics_to_minimize = [metric.name for metric in metrics if metric.direction == Direction.MINIMIZE]
    metrics_to_maximize = [metric.name for metric in metrics if metric.direction == Direction.MAXIMIZE]


    def find_best_model(df: DataFrame) -> str:
        """
        Retrieves the model that leads in the greatest number of metrics

        Args:
            df: dataframe with the scores of each model

        Returns: name of the model
        """
        count_min = ((df[metrics_to_minimize] == df[metrics_to_minimize].min()).sum(axis=1))
        count_max = ((df[metrics_to_maximize] == df[metrics_to_maximize].max()).sum(axis=1))
        return (count_min + count_max).idxmax()

    def move_n_rename_folders() -> Tuple[List[str], str]:
        """
        Moves the experiment folders into the proper directory and renames them.

        Returns: List of keywords that identify the experiment, name of the new directory
        """
        # We extract the folders name associated to the completed experiments
        experiment_folders = [f for f in listdir(Paths.EXPERIMENTS_RECORDS) if f not in ['.gitkeep', OB, VO2]]

        # We extract the keywords that help identify the experiment
        keywords = experiment_folders[0].split('_')

        # We create of a new subfolder associated to the task and specific keywords
        new_directory = join(result_folder, '_'.join(keywords[2:]))
        makedirs(result_subfolder)

        # We move the folders into the right directory
        for f in experiment_folders:
            move(join(Paths.EXPERIMENTS_RECORDS, f), new_directory)

        # We rename experiment folders
        for f in listdir(new_directory):
            rename(join(new_directory, f), join(new_directory, f.split('_')[0]))

        return keywords, new_directory

    """
    1. Data preparation
    """
    # If we want to use the generated data
    add_delimiter("1. Data preparation")
    if args.data == GENERATED:

        # We create the learning set and the holdout set if they don't already exists
        if not exists(learning_set_path) or not exists(holdout_set_path):
            script_path = join(Paths.UTILS_SCRIPTS, "generate_experiment_tables.py")
            csv_path = join(Paths.DATA, f"{args.task}_dataset.csv")
            script_args = ['-csv', csv_path, '-tc', DUMMY, '-cat', '-nt', args.task.upper()]
            check_call(args=['python', script_path, *script_args])

        # We create the mask for the stratified random sampling
        if not exists(mask_path):
            script_path = join(Paths.UTILS_SCRIPTS, "generate_masks.py")
            script_args = ['-csv', learning_set_path, '-tc', DUMMY, '-cat', '-fn', f"{args.task}_mask"]
            check_call(args=['python', script_path, *script_args])

    """
    2.1 Evaluation of models with manually selected hyperparameters
    """
    add_delimiter("2.1 Evaluation of models - Manual")
    script_path = join(Paths.EXPERIMENTS_SCRIPTS, args.task, 'manual_evaluations.py')
    graph_args = ['-deg'] + [str(2 * i) for i in range(4, 6)] + ['-cond_col']
    check_call(args=['python', script_path, *feature_args, *model_args, '-gcn', '-gat', *graph_args])

    """
    2.2 Results compilation
    """
    add_delimiter("2.2 Results compilation - Manual")

    # We create of a new directory specific to the task
    result_folder = join(Paths.EXPERIMENTS_RECORDS, args.task)
    makedirs(result_folder, exist_ok=True)

    # We move and rename folders that are within the experiment records directory
    keywords, result_subfolder = move_n_rename_folders()

    # We create a csv with the scores
    script_path = join(Paths.UTILS_SCRIPTS, 'get_scores_csv.py')
    csv_name = '_'.join(keywords[1:])
    check_call(args=['python', script_path, '-p', result_subfolder, '-fn', csv_name])

    # We extract the best observed degrees associated to each GNNs
    df = read_csv(join(Paths.CSV_FILES, f"{csv_name}.csv"), index_col=0)
    gat_k = extract_best_graph_degree(df, GAT)
    gcn_k = extract_best_graph_degree(df, GAT)

    """
    3.1 Evaluation of models with automated hyperparameter optimization
    """
    add_delimiter("3.1 Evaluation of models - Automated")
    script_path = join(Paths.EXPERIMENTS_SCRIPTS, args.task, 'automated_evaluations.py')
    check_call(args=['python', script_path, *feature_args, *model_args])
    check_call(args=['python', script_path, *feature_args, '-gat', '-deg', str(gat_k), '-cond_col'])
    check_call(args=['python', script_path, *feature_args, '-gcn', '-deg', str(gcn_k), '-cond_col'])

    # We move and rename folders that are within the experiment records directory
    keywords, result_subfolder = move_n_rename_folders()

    # We create a csv with the scores
    script_path = join(Paths.UTILS_SCRIPTS, 'get_scores_csv.py')
    csv_name = '_'.join(keywords[1:])
    check_call(args=['python', script_path, '-p', result_subfolder, '-fn', csv_name])