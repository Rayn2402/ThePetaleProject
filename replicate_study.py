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
from webbrowser import open_new_tab

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

# MODEL CONVERSION DICTIONARY
MODEL_TO_ARGS = {'RandomForest': '-rf',
                 'XGBoost': '-xg',
                 'enet': '-enet',
                 'MLP': '-mlp',
                 'ggaeEnet': '-ggae'}


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


def summarize_experiments(result_directory: str, final_results: bool = False) -> str:
    """
    Moves and renames experiment folders, then summarize results into csv and html files

    Args:
        result_directory: path of the directory created to store the results of the experiments
        final_results: if true, folders will be moved and renamed considering that they contain results
                       from the final tests on the holdout set

    Returns: path of the csv created
    """
    # We extract the folders name associated to the completed experiments
    experiment_folders = [f for f in listdir(Paths.EXPERIMENTS_RECORDS) if f not in ['.gitkeep', OB, VO2]]

    # We extract the keywords that help identify the experiment
    keywords = experiment_folders[0].split('_')[1:]

    # We create of a new subfolder associated to the task and specific keywords
    if not final_results:
        new_directory = join(result_directory, '_'.join(keywords[1:]))
    else:
        new_directory = join(result_directory, '_'.join(keywords[2:] + ['holdout']))

    makedirs(new_directory)

    # We move the folders into the right directory
    for f in experiment_folders:
        move(join(Paths.EXPERIMENTS_RECORDS, f), new_directory)

    # We rename experiment folders
    for f in listdir(new_directory):

        if not final_results:
            new_name = f.split('_')[0]

        else:
            keys_list = f.split('_')
            new_name = '_'.join([keys_list[0], keys_list[2]])

        rename(join(new_directory, f), join(new_directory, new_name))

    # We create a csv summarizing scores of all models
    csv_name = '_'.join(keywords) if not final_results else '_'.join([keywords[0]] + keywords[2:] + ['holdout'])
    check_call(args=['python', join(Paths.UTILS_SCRIPTS, 'get_scores_csv.py'),
                     '-p', new_directory, '-fn', csv_name])

    # We create an interactive html file summarizing the results of all models
    check_call(args=['python', join(Paths.UTILS_SCRIPTS, 'create_experiments_recap.py'),
                     '-p', new_directory, '-fn', 'recap'])

    # We open the html file
    open_new_tab(join(new_directory, 'recap.html'))

    # We return the path of the csv
    return join(Paths.CSV_FILES, f'{csv_name}.csv')


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


if __name__ == '__main__':

    args = argument_parser()

    """
    0. Setup
    """
    # We set main variables according to the prediction task
    metrics = [AbsoluteError(), ConcordanceIndex(), Pearson(), RootMeanSquaredError()]
    model_args = ['-xg']

    if args.task == OB:
        learning_set_path = Paths.OBESITY_LEARNING_SET_CSV
        holdout_set_path = Paths.OBESITY_HOLDOUT_SET_CSV
        eval_mask_path = Paths.OBESITY_MASK
        holdout_mask_path = Paths.OBESITY_HOLDOUT_MASK
        feature_args = ['-b', '-f']
    else:
        learning_set_path = Paths.VO2_LEARNING_SET_CSV
        holdout_set_path = Paths.VO2_HOLDOUT_SET_CSV
        eval_mask_path = Paths.VO2_MASK
        holdout_mask_path = Paths.VO2_HOLDOUT_MASK
        feature_args = ['-b', '-s', '-r_w']
        #metrics += [Sensitivity(), Specificity()]

    to_minimize = [metric.name for metric in metrics if metric.direction == Direction.MINIMIZE]
    to_maximize = [metric.name for metric in metrics if metric.direction == Direction.MAXIMIZE]

    # We create of a new directory specific to the task
    result_folder = join(Paths.EXPERIMENTS_RECORDS, args.task)
    makedirs(result_folder, exist_ok=True)

    # Creation of a function specific to the metrics associated with the task
    def find_best_model(dataframe: DataFrame) -> str:
        """
        Retrieves the model that leads in the greatest number of metrics

        Args:
            dataframe: dataframe with the scores of each model

        Returns: name of the model
        """
        count_min = ((dataframe[to_minimize] == dataframe[to_minimize].min()).sum(axis=1))
        count_max = ((dataframe[to_maximize] == dataframe[to_maximize].max()).sum(axis=1))
        return (count_min + count_max).idxmax()

    """
    1. Data preparation
    """
    # If we want to use the generated data
    add_delimiter("1. Data preparation")
    if args.data == GENERATED:

        # We create the learning set and the holdout set if they don't already exists
        if not exists(learning_set_path) or not exists(holdout_set_path):
            csv_path = join(Paths.DATA, f"{args.task}_dataset.csv")
            script_args = ['-csv', csv_path, '-tc', DUMMY, '-cat', '-nt', args.task.upper()]
            check_call(args=['python', join(Paths.UTILS_SCRIPTS, "generate_experiment_tables.py"), *script_args])

        # We create the stratified random sampling masks for the evaluation of models
        if not exists(eval_mask_path):
            script_args = ['-csv', learning_set_path, '-tc', DUMMY, '-cat', '-fn', f"{args.task}_mask"]
            check_call(args=['python', join(Paths.UTILS_SCRIPTS, "generate_masks.py"), *script_args])

        # We create the stratified random sampling masks for the final tests
        if not exists(holdout_mask_path):
            check_call(args=['python', join(Paths.EXPERIMENTS_SCRIPTS, args.task, 'holdout_mask_creation.py'),
                             '-from_csv'])

    """
    2.1 Evaluation of models with manually selected hyperparameters
    """
    add_delimiter("2.1 Evaluation of models - Manual")
    manual_script_path = join(Paths.EXPERIMENTS_SCRIPTS, args.task, 'manual_evaluations.py')
    graph_args = ['-deg'] + [str(2 * i) for i in range(5, 6)] + ['-cond_col']
    check_call(args=['python', manual_script_path, *feature_args, *model_args, '-gcn', '-gat', *graph_args,
                     '-k', '2', '-l', '2'])

    """
    2.2 Results compilation
    """
    add_delimiter("2.2 Results compilation - Manual")

    # We compile and summarize results
    manual_scores_csv = summarize_experiments(result_folder)

    # We extract the best observed degrees associated to each GNNs
    df = read_csv(manual_scores_csv, index_col=0)
    gat_k = extract_best_graph_degree(df, GAT)
    gcn_k = extract_best_graph_degree(df, GAT)

    """
    3.1 Evaluation of models with automated hyperparameter optimization
    """
    add_delimiter("3.1 Evaluation of models - Automated")
    automated_script_path = join(Paths.EXPERIMENTS_SCRIPTS, args.task, 'automated_evaluations.py')
    check_call(args=['python', automated_script_path, *feature_args, *model_args, '-k', '2', '-l', '2'])
    check_call(args=['python', automated_script_path, *feature_args, '-gat', '-deg', str(gat_k), '-cond_col',
                     '-k', '2', '-l', '2'])
    check_call(args=['python', automated_script_path, *feature_args, '-gcn', '-deg', str(gcn_k), '-cond_col',
                     '-k', '2', '-l', '2'])

    """
    3.2 Results compilation - Automated
    """
    add_delimiter("3.2 Results compilation - Automated")

    # We compile and summarize results
    automated_scores_csv = summarize_experiments(result_folder)

    """
    4. Selection of the best model
    """
    add_delimiter("4. Selection of the best model")

    # We load the scores obtained during the manual and the automated evaluation
    df = read_csv(manual_scores_csv, index_col=0)
    df = df.append(read_csv(automated_scores_csv, index_col=0))

    # We find the best model for the final test
    best_model = find_best_model(df)
    print(f"Best model : {best_model}")

    # We infer the correct arguments for the final test
    if GCN in best_model or GAT in best_model:
        arguments = [f'-{best_model[0:2].lower()}', '-deg', str(best_model[3:]), '-cond_col', '-holdout']

    else:
        arguments = [MODEL_TO_ARGS[best_model], '-holdout']

    """
    5. Final tests
    """
    add_delimiter("5.1 Final test - Manual")
    check_call(args=['python', manual_script_path, *feature_args, *arguments, '-k', '2', '-l', '2'])

    add_delimiter("5.2 Final test - Automated")
    check_call(args=['python', automated_script_path, *feature_args, *arguments, '-k', '2', '-l', '2'])

    """
    6. Final results compilation
    """

    add_delimiter("5.3 Final results compilation")

    # We compile and summarize results
    _ = summarize_experiments(result_folder, final_results=True)






