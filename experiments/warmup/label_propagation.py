"""
Filename: label_propagation.py

Authors: Nicolas Raymond

Description: This file is used to to experiment the
             label propagation method using warmup dataset

Date of last modification : 2021/11/08
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

    # Nb inner split and nb outer split selection
    parser.add_argument('-p', '--path', type=str,
                        help='Path of the folder from which to take predictions.'
                             '(Ex. records/experiments/warmup/linear_reg')
    parser.add_argument('-nb_iter', '--nb_iter', type=int,
                        help='Number of correction adn smoothing iterations')
    parser.add_argument('-r', '--restart_proba', type=float,
                        help='Restart probability used in propagation algorithm')
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
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.sampling import extract_masks, GeneChoice, get_warmup_data, MaskType, push_valid_to_train
    from src.data.extraction.data_management import PetaleDataManager
    from src.recording.recording import Recorder, compare_prediction_recordings, get_evaluation_recap
    from src.utils.graph import PetaleGraph, correct_and_smooth
    from src.utils.score_metrics import AbsoluteError, Pearson, SquaredError, RootMeanSquaredError

    # Arguments parsing
    args = argument_parser()

    # Extraction of masks
    masks = extract_masks(Paths.WARMUP_MASK, k=5, l=0)
    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [AbsoluteError(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Extraction of data
    manager = PetaleDataManager()
    df, target, cont_cols, cat_cols = get_warmup_data(manager, baselines=True,
                                                      genes=GeneChoice.SIGNIFICANT, sex=True)

    # Creation of dataset
    dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False, to_tensor=True)

    # Evaluation name saving
    evaluation_name = "warmup_label_propagation"

    # We extract data
    _, y,  _ = dataset[:]

    # For all splits
    for k, m in tqdm(masks.items()):

        # We extract the records from folder "Split k"
        with open(join(args.path, f"Split_{k}", "records.json")) as json_file:
            records_k = json.load(json_file)

        # We save the predictions made for each id
        pred = zeros((len(dataset), 1))
        for id_, result_dict in records_k["train_results"].items():
            pred[dataset.ids_to_row_idx[id_]] = float(result_dict['prediction'])

        for id_, result_dict in records_k["test_results"].items():
            pred[dataset.ids_to_row_idx[id_]] = float(result_dict['prediction'])

        # We update dataset mask
        dataset.update_masks(train_mask=m[MaskType.TRAIN], test_mask=m[MaskType.TEST])

        # Recorder initialization
        recorder = Recorder(evaluation_name=evaluation_name,
                            index=k, recordings_path=Paths.EXPERIMENTS_RECORDS)

        # We build the graph
        g = PetaleGraph(dataset, include_distances=args.include_distances,
                        cat_cols=dataset.cat_cols, max_degree=args.max_degree)

        # We proceed to correction and smoothing of the predictions
        y_copy = deepcopy(y)
        cs_pred = correct_and_smooth(g, pred, y_copy, m, r=args.restart_proba, nb_iter=args.nb_iter)

        for mask, test in [(m[MaskType.TRAIN], False), (m[MaskType.TEST], True)]:

            # We record predictions
            pred, ground_truth = cs_pred[mask], y[mask]
            recorder.record_predictions([dataset.ids[i] for i in mask], pred, ground_truth, test)

            # We record scores
            for metric in evaluation_metrics:
                recorder.record_scores(score=metric(pred, ground_truth), metric=metric.name, test=test)

        # Generation of the file with the results
        recorder.generate_file()
        compare_prediction_recordings(evaluations=[evaluation_name], split_index=k,
                                      recording_path=Paths.EXPERIMENTS_RECORDS)

    get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)
