"""
Filename: meta_learner_experiment.py

Authors: Nicolas Raymond

Description: This file is used to store the meta learner experiment.
             We first train a linear regression, make a prediction on all the data points and
             register scores on the test set.

             We then train an heterogeneous graph attention network using only the predictions
             of the linear prediction as features.

Date of last modification : 2021/11/08
"""

import argparse
import sys
from copy import deepcopy
from json import load
from numpy import array
from os.path import dirname, realpath, join


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python meta_learner_experiment.py -p [path]',
                                     description="Loads the prediction of a model over the different"
                                                 "splits of the warmup dataset and train a HAN meta-learner"
                                                 "using them.")

    # Nb inner split and nb outer split selection
    parser.add_argument('-p', '--path', type=str,
                        help='Number of outer splits during the models evaluations')

    # Evaluation name
    parser.add_argument('-e_name', '--eval_name', type=str,
                        help='Name of the evaluation')

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
    from src.data.extraction.constants import PARTICIPANT
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleStaticGNNDataset
    from src.data.processing.sampling import extract_masks, GeneChoice, get_warmup_data, MaskType, push_valid_to_train
    from src.models.han import PetaleHANR
    from src.recording.constants import PREDICTION, TARGET, TEST_RESULTS, TRAIN_RESULTS, RECORDS_FILE
    from src.recording.recording import compare_prediction_recordings, get_evaluation_recap, Recorder
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError, SquaredError

    # Argument parsing
    args = argument_parser()
    print("Warning - The predictions must come from a model that trained without a valid set\n")

    # Initialization of DataManager and sampler
    manager = PetaleDataManager()

    # Extraction of the mask
    masks = extract_masks(Paths.WARMUP_MASK, k=10, l=1)

    # Creation of mask copy with no valid
    no_valid_masks = deepcopy(masks)
    push_valid_to_train(no_valid_masks)

    # Metrics
    metrics = [AbsoluteError(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Extraction of data
    df, target, cont_cols, cat_cols = get_warmup_data(manager, genes=GeneChoice.SIGNIFICANT)

    # We remove the continuous columns
    df.drop(cont_cols, inplace=True, axis=1)

    # We save the name of the column that will store the predictions
    cont_cols = ['feature_1']

    # We save a map of participant id to their row index in the df
    id_to_idx_map = {id_: i for i, id_ in enumerate(df[PARTICIPANT].values)}

    """
    Training and evaluation of meta learner (HAN) over the different split
    """
    # For each mask
    for k, m in masks.items():

        # Recorder initialization
        recorder = Recorder(evaluation_name=args.eval_name,
                            index=k,
                            recordings_path=Paths.EXPERIMENTS_RECORDS)

        # Extraction of train and test mask
        train_mask, valid_mask, test_mask = m[MaskType.TRAIN], m[MaskType.VALID], m[MaskType.TEST]

        # Loading of predictions (training predictions include people in the valid set)
        with open(join(args.path, f"Split_{k}", RECORDS_FILE), "r") as read_file:
            data = load(read_file)

        train_pred = {p_id: round(float(v[PREDICTION]), 2) for p_id, v in data[TRAIN_RESULTS].items()}
        train_ground_truth = [float(v[TARGET]) for v in data[TRAIN_RESULTS].values()]
        test_pred = {p_id: round(float(v[PREDICTION]), 2) for p_id, v in data[TEST_RESULTS].items()}
        unordered_pred = list(train_pred.values()) + list(test_pred.values())

        # Identification of idx related to participant
        participant_id = list(train_pred.keys()) + list(test_pred.keys())

        # Sorting of predictions
        pred = [0]*len(unordered_pred)
        for i, id_ in enumerate(participant_id):
            pred[id_to_idx_map[id_]] = unordered_pred[i]

        # Creation of the dataset
        df[cont_cols[0]] = array(pred)
        han_dataset = PetaleStaticGNNDataset(df, target, cont_cols, cat_cols, classification=False)
        han_dataset.update_masks(train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)

        # Training of meta learner
        han_reg = PetaleHANR(meta_paths=han_dataset.get_metapaths(),
                             hidden_size=10,
                             rho=0,
                             num_heads=20,
                             cat_idx=han_dataset.cat_idx,
                             cat_sizes=han_dataset.cat_sizes,
                             cat_emb_sizes=han_dataset.cat_sizes,
                             num_cont_col=1,
                             batch_size=15,
                             patience=25,
                             eval_metric=metrics[-1])

        han_reg.fit(han_dataset)

        # Creation of dataset where training set 'feature_1' value is the ground truth
        unordered_pred = train_ground_truth + list(test_pred.values())
        pred = [0] * len(unordered_pred)
        for i, id_ in enumerate(participant_id):
            pred[id_to_idx_map[id_]] = unordered_pred[i]
        df[cont_cols[0]] = pred
        han_dataset = PetaleStaticGNNDataset(df, target, cont_cols, cat_cols, classification=False)
        han_dataset.update_masks(train_mask=no_valid_masks[k][MaskType.TRAIN],
                                 test_mask=no_valid_masks[k][MaskType.TEST])

        # Prediction calculations and recording
        for mask, test in [(train_mask, False), (test_mask, True)]:

            _, y, _ = han_dataset[mask]
            predictions = han_reg.predict(dataset=han_dataset, mask=mask)
            recorder.record_predictions([han_dataset.ids[i] for i in mask], predictions, y, test)

            # Score calculation and recording
            for metric in metrics:
                recorder.record_scores(score=metric(predictions, y), metric=metric.name, test=test)

        # Generation of the file with the results
        recorder.generate_file()
        compare_prediction_recordings(evaluations=[args.eval_name], split_index=k,
                                      recording_path=Paths.EXPERIMENTS_RECORDS)

    get_evaluation_recap(evaluation_name=args.eval_name, recordings_path=Paths.EXPERIMENTS_RECORDS)

