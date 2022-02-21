"""
Filename: label_propagation.py

Authors: Nicolas Raymond

Description: This file is used to to experiment the
             label propagation method using warmup dataset

Date of last modification : 2022/02/21
"""

import argparse
import json
import sys

from copy import deepcopy
from os.path import dirname, realpath, join
from torch import zeros
from tqdm import tqdm


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python label_propagation.py',
                                     description="Runs the correction and smoothing experiment"
                                                 " associated to the warmup dataset")

    # Parameters selection
    parser.add_argument('-p', '--path', type=str,
                        help='Path of the folder from which to take predictions.'
                             '(Ex. records/experiments/warmup/enet...')
    parser.add_argument('-nb_iter', '--nb_iter', type=int,
                        help='Number of correction adn smoothing iterations')
    parser.add_argument('-rc', '--r_correct', type=float,
                        help='Restart probability used in propagation algorithm for correction')
    parser.add_argument('-rs', '--r_smooth', type=float,
                        help='Restart probability used in propagation algorithm for smoothing')
    parser.add_argument('-d', '--include_distances', default=False, action='store_true',
                        help='True if to consider distances during graph construction')
    parser.add_argument('-max_degree', '--max_degree', type=int,
                        help='Maximum degree of a nodes in the graph')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.processing.datasets import MaskType, PetaleDataset
    from src.data.processing.sampling import extract_masks, GeneChoice, get_warmup_data, MaskType, push_valid_to_train
    from src.data.extraction.data_management import PetaleDataManager
    from src.recording.constants import PREDICTION, RECORDS_FILE, TRAIN_RESULTS, TEST_RESULTS
    from src.recording.recording import Recorder, compare_prediction_recordings, get_evaluation_recap
    from src.utils.graph import PetaleGraph, correct_and_smooth
    from src.utils.score_metrics import AbsoluteError, Pearson, SquaredError, RootMeanSquaredError

    # Arguments parsing
    args = argument_parser()

    # Extraction of masks
    masks = extract_masks(Paths.WARMUP_MASK, k=10, l=0)
    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [AbsoluteError(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Extraction of data
    manager = PetaleDataManager()
    df, target, cont_cols, cat_cols = get_warmup_data(manager, baselines=True, sex=True)

    # Creation of dataset
    dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False, to_tensor=True)

    # Evaluation name saving
    evaluation_name = "warmup_label_propagation"

    # For all splits
    for k, m in tqdm(masks.items()):

        # We extract the records from folder "Split k"
        with open(join(args.path, f"Split_{k}", RECORDS_FILE)) as json_file:
            records_k = json.load(json_file)

        # We save the predictions made for each id
        pred = zeros((len(dataset), 1))
        for id_, result_dict in records_k[TRAIN_RESULTS].items():
            pred[dataset.ids_to_row_idx[id_]] = float(result_dict[PREDICTION])

        for id_, result_dict in records_k[TEST_RESULTS].items():
            pred[dataset.ids_to_row_idx[id_]] = float(result_dict[PREDICTION])

        # We update dataset mask
        dataset.update_masks(train_mask=m[MaskType.TRAIN], test_mask=m[MaskType.TEST])

        # Recorder initialization
        recorder = Recorder(evaluation_name=evaluation_name,
                            index=k, recordings_path=Paths.EXPERIMENTS_RECORDS)

        # We build the graph
        g = PetaleGraph(dataset, include_distances=args.include_distances,
                        cat_cols=dataset.cat_cols, max_degree=args.max_degree)

        # We proceed to correction and smoothing of the predictions
        y_copy = deepcopy(dataset.y)
        cs_pred = correct_and_smooth(g, pred=pred, labels=y_copy, masks=m, r_correct=args.r_correct,
                                     r_smooth=args.r_smooth, nb_iter=args.nb_iter)

        for mask, masktype in [(m[MaskType.TRAIN], MaskType.TRAIN), (m[MaskType.TEST], MaskType.TEST)]:

            # We record predictions
            pred, ground_truth = cs_pred[mask], dataset.y[mask]
            recorder.record_predictions([dataset.ids[i] for i in mask], pred, ground_truth, mask_type=masktype)

            # We record scores
            for metric in evaluation_metrics:
                recorder.record_scores(score=metric(pred, ground_truth), metric=metric.name, mask_type=masktype)

        # Generation of the file with the results
        recorder.generate_file()
        compare_prediction_recordings(evaluations=[evaluation_name], split_index=k,
                                      recording_path=Paths.EXPERIMENTS_RECORDS)

    get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)
