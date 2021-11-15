"""
Filename: graph_collaborative_filtering.py

Authors: Nicolas Raymond

Description: This file is used to execute the graph collaborative
             filtering experiment on warmup dataset

Date of last modification : 2021/11/08
"""

import sys

from os.path import dirname, realpath
from tqdm import tqdm


if __name__ == '__main__':

    # Imports related to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.sampling import extract_masks, GeneChoice, get_warmup_data, MaskType
    from src.data.processing.datasets import PetaleDataset
    from src.utils.graph import PetaleGraph
    from src.utils.collaborative_filtering import run_collaborative_filtering
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError, SquaredError
    from src.recording.recording import Recorder, get_evaluation_recap, compare_prediction_recordings
    from settings.paths import Paths

    # Generation of dataset
    data_manager = PetaleDataManager()
    df, target, cont_cols, cat_cols = get_warmup_data(data_manager, genes=GeneChoice.SIGNIFICANT, sex=True)

    # Creation of the dataset
    dataset = PetaleDataset(df, target, cont_cols, cat_cols=cat_cols,
                            classification=False, to_tensor=True)

    # Extraction of labels
    _, y, _ = dataset[:]

    # Extraction of masks
    masks = extract_masks(Paths.WARMUP_MASK, k=10, l=0)

    # We save the evaluation metrics
    eval_metrics = [AbsoluteError(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Creation of the graph
    g = PetaleGraph(dataset)
    print("Graph done")

    # Running of random walk
    weights = g.random_walk_with_restart(1, .10)
    print("Random walk done")

    # Saving of evaluation name
    evaluation_name = 'warmup_collaborative_filtering'

    # We test collaborative filtering on each split
    for k, v in tqdm(masks.items()):

        # Masks extraction and dataset update
        train_mask, test_mask = v[MaskType.TRAIN], v[MaskType.TEST]

        # Recorder initialization
        recorder = Recorder(evaluation_name=evaluation_name,
                            index=k, recordings_path=Paths.EXPERIMENTS_RECORDS)

        # We run collaborative filtering
        pred = run_collaborative_filtering(weights, y, test_mask, top_k=3)

        for mask, test in [(train_mask, False), (test_mask, True)]:

            # Predictions recording
            recorder.record_predictions([dataset.ids[i] for i in mask], pred[mask], y[mask], test)

            # Score calculation and recording
            for metric in eval_metrics:
                recorder.record_scores(score=metric(pred[mask], y[mask]), metric=metric.name, test=test)

        # Generation of the file with the results
        recorder.generate_file()
        compare_prediction_recordings(evaluations=[evaluation_name], split_index=k,
                                      recording_path=Paths.EXPERIMENTS_RECORDS)

    get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)