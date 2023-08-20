"""
Filename: original_equation.py

Authors: Nicolas Raymond

Description: This file is used to evaluate the original equation on the VO2 peak dataset.

Date of last modification : 2022/07/07
"""

import argparse

from os.path import dirname, realpath
import sys

sys.path.append((dirname(dirname(dirname(realpath(__file__))))))
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.datasets import MaskType, PetaleDataset
from src.data.processing.transforms import ContinuousTransform
from src.data.processing.sampling import extract_masks, get_VO2_data
from src.utils.metrics import AbsoluteError, Pearson, RootMeanSquaredError, SquaredError, SpearmanR, TopKAbsoluteError
from src.data.extraction.constants import *
from src.recording.recording import compare_prediction_recordings, get_evaluation_recap, Recorder
from settings.paths import Paths
from torch import tensor
from typing import Dict, List


# We create the function that will calculate the VO2 peak values based on the original equation
def original_equation(item):
    return -0.236 * item[AGE] - 0.094 * item[WEIGHT] - 0.120 * item[TDM6_HR_END] + 0.067 * item[TDM6_DIST] + \
           0.065 * item[MVLPA] - 0.204 * item[DT] + 25.145


def get_arguments():
    """
    This function defines a parser for the original equation experiment
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python original_equation.py',
                                     description="Runs the original equation experiment")

    parser.add_argument('-k', '--nb_outer_splits', type=int, default=5,
                        help="Number of outer splits (default = 5)")

    return parser.parse_args()


def execute_original_equation_experiment(dts: PetaleDataset,
                                         m: Dict[int, Dict[str, List[int]]],
                                         eval_name: str) -> None:
    """
    Executes the original equation experiment

    Args:
        dts:  dataset with inputs and regression targets
        m: dictionary with list of idx to use for training and testing
        eval_name: name of the results file saved at the recordings_path


    Returns: None
    """
    # We save the evaluation metrics
    eval_metrics = [AbsoluteError(), SpearmanR(), Pearson(), SquaredError(),
                    TopKAbsoluteError(), RootMeanSquaredError()]

    # We run tests for each masks
    for k, v in m.items():

        # Masks extraction and dataset update
        train_mask, test_mask = v[MaskType.TRAIN], v[MaskType.TEST]
        dts.update_masks(train_mask=train_mask, test_mask=test_mask)

        # Data extraction and preprocessing without normalization
        x_copy = dts.original_data.copy()
        mu, _, _ = dts.current_train_stats()
        x_copy = ContinuousTransform.fill_missing(x_copy, mu)

        # Recorder initialization
        recorder = Recorder(evaluation_name=eval_name,
                            index=k, recordings_path=Paths.EXPERIMENTS_RECORDS)

        # Prediction calculations and recording
        for mask, m_type in [(train_mask, MaskType.TRAIN), (test_mask, MaskType.TEST)]:

            # Data extraction
            x = x_copy.iloc[mask]
            _, y, _ = dts[mask]

            # Prediction
            predictions = []
            for i, row in x.iterrows():
                predictions.append(original_equation(row))

            predictions = tensor(predictions)
            recorder.record_predictions([dataset.ids[i] for i in mask], predictions, y, mask_type=m_type)

            # Score calculation and recording
            for metric in eval_metrics:
                recorder.record_scores(score=metric(predictions, y), metric=metric.name, mask_type=m_type)

        # Generation of the file with the results
        recorder.generate_file()
        compare_prediction_recordings(evaluations=[eval_name], split_index=k,
                                      recording_path=Paths.EXPERIMENTS_RECORDS)

    get_evaluation_recap(evaluation_name=eval_name, recordings_path=Paths.EXPERIMENTS_RECORDS)


if __name__ == '__main__':

    # Arguments parsing
    args = get_arguments()

    # Arguments extraction
    k = args.nb_outer_splits

    # Generation of dataset
    df, target, cont_cols, cat_cols = get_VO2_data(PetaleDataManager())

    # Creation of the dataset
    dataset = PetaleDataset(df, target, cont_cols, cat_cols=cat_cols,
                            classification=False, to_tensor=True)

    # Extraction of masks
    masks = extract_masks(Paths.VO2_MASK, k=k, l=0)
    evaluation_name = f"original_equation"

    # Execution of the experiment
    execute_original_equation_experiment(dts=dataset, m=masks, eval_name=evaluation_name)
