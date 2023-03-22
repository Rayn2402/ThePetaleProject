"""
Filename: results_analysis.py

Author: Nicolas Raymond

Description: Contains function to help analyses results from different experiments

Date of last modification: 2022/04/27
"""

from json import dump, load
from numpy import mean, std
from os import listdir
from os.path import join, isdir
from pandas import DataFrame, merge, read_csv
from settings.paths import Paths
from src.data.extraction.constants import *
from src.data.extraction.data_management import PetaleDataManager
from src.recording.recording import get_evaluation_recap, Recorder
from src.utils.metrics import Sensitivity, Specificity, BinaryBalancedAccuracy
from time import time
from torch import tensor
from typing import Any, Callable, Dict, List, Optional, Union

APRIORI_KEYS = ['Support', 'Lift', 'Confidence']
SECTION = 'Section'
CLASS_PRED = 'CP'
REG_PRED = 'RP'


def extract_predictions(paths: List[str],
                        model_ids: List[str],
                        filename: str) -> None:
    """
    Extracts the predictions of different models and store them in a csv file

    Args:
        paths: list of paths where the records are stored
        model_ids: list of names to identify models from which the predictions are retrieved
        filename: name of the file in which the predictions will be stored

    Returns: None
    """
    predictions = {}
    for i, m, p in zip(range(len(model_ids)), model_ids, paths):

        # We load the data from the records
        with open(join(p, Recorder.RECORDS_FILE), "r") as read_file:
            data = load(read_file)

        if i == 0:
            for k, v in data[Recorder.TEST_RESULTS].items():
                predictions[k] = {Recorder.TARGET: v[Recorder.TARGET], m: v[Recorder.PREDICTION]}
        else:
            for k, v in data[Recorder.TEST_RESULTS].items():
                predictions[k][m] = v[Recorder.PREDICTION]

    # We use the dict to create a dataframe
    df = DataFrame.from_dict(data=predictions, orient='index')
    df.sort_values(Recorder.TARGET, inplace=True)

    # We save the dataframe in a csv
    df.to_csv(path_or_buf=f"{filename}.csv")


def get_directories(path: str) -> List[str]:
    """
    Extracts the names of all the folders that can be found at the given path

    Args:
        path: directory

    Returns: list of folder names (str)
    """
    return [folder for folder in listdir(path) if isdir(join(path, folder))]


def get_experiment_summaries(path: str,
                             csv_filename: str,
                             nb_digits: int = 2) -> None:
    """
    From a folder containing experiment results of different models,
    we extract test metrics and save it in a csv that can then be converted to a LaTeX table using :

    https://www.tablesgenerator.com/

    Args:
        path: directory where the experiment results are stored
        csv_filename: name of the csv file containing the extracted data
        nb_digits: number of digits kept in the csv

    Returns: None
    """
    # We extract the names of all the folders in the directory
    folders = get_directories(path)

    # We extract test metrics data from each model directory
    test_metrics = {}
    for f in folders:

        # We extract the name of the model
        m = f.split('_')[0]

        # We load the data from the summary
        with open(join(path, f, Recorder.SUMMARY_FILE), "r") as read_file:
            data = load(read_file)

        # We save the metrics in the dictionary
        test_metrics[m] = {metric: f"{round(float(v[Recorder.MEAN]), nb_digits)} +- {round(float(v[Recorder.STD]), nb_digits)}"
                           for metric, v in data[Recorder.TEST_METRICS].items()}

    # We store the data in a dataframe and save it into a csv
    df = DataFrame(data=test_metrics).T
    df.to_csv(path_or_buf=join(Paths.CSV_FILES, f"{csv_filename}.csv"))






