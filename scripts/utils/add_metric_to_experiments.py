"""
Filename: add_metric_to_experiment.py

Author: Nicolas Raymond

Description: script used to add a missing metric to an experiment

Date of last modification: -
"""

import sys

from argparse import ArgumentParser
from json import dump, load
from os.path import dirname, join, realpath
from torch import tensor

# Imports specific to project
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from src.recording.recording import get_evaluation_recap, Recorder
from src.utils.results_analyses import get_directories
from src.utils.metrics import MeanAbsolutePercentageError


def argument_parser():
    """
    This function defines a parser that to extract scores of each model within an experiment
    """
    # Create a parser
    parser = ArgumentParser(usage='\n python add_metric_to_experiment.py [experiment_folder_paths]',
                            description="Add a missing metric to all the recorded files within multiple experiments")

    parser.add_argument('-p', '--paths', nargs='*', type=str, help='Paths of the experiment folders')

    arguments = parser.parse_args()

    return arguments


METRIC = MeanAbsolutePercentageError()

if __name__ == '__main__':

    # Extract argument
    args = argument_parser()

    for p in args.paths:

        # For each experiment split folder
        for folder in get_directories(p):

            # Load records file
            with open(join(p, folder, Recorder.RECORDS_FILE), "r") as read_file:
                data = load(read_file)

            # Add the metric score for each section
            for s1, s2 in [(Recorder.TRAIN_RESULTS, Recorder.TRAIN_METRICS),
                           (Recorder.TEST_RESULTS, Recorder.TEST_METRICS),
                           (Recorder.VALID_RESULTS, Recorder.VALID_METRICS)]:

                if data.get(s1) is not None:
                    pred, targets = [], []
                    for k in data[s1].keys():
                        pred.append(float(data[s1][k][Recorder.PREDICTION]))
                        targets.append(float(data[s1][k][Recorder.TARGET]))

                    pred, targets = tensor(pred), tensor(targets)
                    data[s2][METRIC.name] = METRIC(pred=pred, targets=targets)

            # Update the records file
            with open(join(p, folder, Recorder.RECORDS_FILE), "w") as file:
                dump(data, file, indent=True)

        # Update the file containing the summary of the experiment
        get_evaluation_recap(evaluation_name='', recordings_path=p)



