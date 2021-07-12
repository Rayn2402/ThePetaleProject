"""


This file is used to store the experiment testing the original equation on the WarmUp dataset.
"""

import argparse
from os.path import dirname, realpath
import sys

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.sampling import TRAIN, TEST, get_warmup_data, extract_masks
from src.data.processing.datasets import PetaleRFDataset
from src.data.processing.transforms import ContinuousTransform, CategoricalTransform
from src.utils.score_metrics import RootMeanSquaredError, AbsoluteError
from src.data.extraction.constants import *
from torch import tensor
from src.recording.recording import Recorder, get_evaluation_recap, compare_prediction_recordings
from settings.paths import Paths
from os.path import join
from typing import Dict, List


# We create the function that will calculate the vo2 Peak value based on the original equation
def original_equation(item):
    return -0.236 * item[AGE] - 0.094 * item[WEIGHT] - 0.120 * item[TDM6_HR_END] + 0.067 * item[TDM6_DIST] + \
           0.065 * item[MVLPA] - 0.204 * item[DT] + 25.145


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 original_equation.py',
                                     description="Runs the original equation experiment")

    parser.add_argument('-nos', '--nb_outer_splits', type=int, default=20,
                        help="Number of outer splits (default = 20)")

    parser.add_argument('-u', '--user', type=str, default='rayn2402',
                        help="Valid username for petale database")

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


def execute_original_equation_experiment(dataset: PetaleRFDataset,
                                         masks: Dict[int, Dict[str, List[int]]],
                                         evaluation_name:str) -> None:
    """
        Function that executes a Neural Network experiments

         Args:
            dataset:  dataset with inputs and regression targets
            masks: dictionary with list of idx to use for training and testing
            evaluation_name: name of the results file saved at the recordings_path


         Returns: None

         """
    # We save the metric object
    metric = RootMeanSquaredError()

    # We save the evaluation metrics
    eval_metrics = {'MAE': AbsoluteError(), 'RMSE': metric}

    # We run training and testing for each masks
    for k, v in masks.items():

        # Masks extraction and dataset update
        train_mask, test_mask = v[TRAIN], v[TEST]
        dataset.update_masks(train_mask=train_mask, test_mask=test_mask)

        # Data extraction and preprocessing without normalization
        x = dataset.original_data.copy()
        mu, _, _ = dataset.current_train_stats()
        x = ContinuousTransform.fill_missing(x, mu)

        # Train and test data extraction
        x_test = x.iloc[test_mask]
        _, y_test = dataset[test_mask]

        # Recorder initialization
        recorder = Recorder(evaluation_name=evaluation_name,
                            index=k, recordings_path=Paths.EXPERIMENTS_RECORDS)

        # Prediction calculations and recording
        predictions = []
        for i, row in x_test.iterrows():
            predictions.append(original_equation(row))
        predictions = tensor(predictions)
        recorder.record_predictions([dataset.ids[i] for i in test_mask], predictions, y_test)

        # Score calculation and recording
        for metric_name, f in eval_metrics.items():
            recorder.record_scores(score=f(predictions, y_test), metric=metric_name)

        # Generation of the file with the results
        recorder.generate_file()
        compare_prediction_recordings(evaluations=[evaluation_name], split_index=k,
                                      recording_path=Paths.EXPERIMENTS_RECORDS)
    get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)


if __name__ == '__main__':

    # Arguments parsing
    args = argument_parser()

    # Arguments extraction
    k = args.nb_outer_splits

    # Generation of dataset
    data_manager = PetaleDataManager(args.user)
    df, target, cont_cols, _ = get_warmup_data(data_manager)

    # Creation of the dataset
    dataset = PetaleRFDataset(df, target, cont_cols, cat_cols=None)

    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "l0_masks.json"), k=k, l=0)

    # Execution of the experiment
    evaluation_name = f"original_equation_k{k}"
    execute_original_equation_experiment(dataset=dataset, masks=masks, evaluation_name=evaluation_name)
