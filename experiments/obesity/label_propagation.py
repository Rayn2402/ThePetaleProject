"""
Filename: label_propagation.py

Authors: Nicolas Raymond

Description: This file is used to to experiment the
             label propagation method using obesity dataset

Date of last modification : 2022/03/16
"""

import sys

from os.path import dirname, realpath


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.sampling import extract_masks, GeneChoice, get_learning_one_data
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.argparsers import correct_and_smooth_parser
    from src.utils.experiments import run_correct_and_smooth_experiment
    from src.utils.score_metrics import AbsoluteError, Pearson, SquaredError, RootMeanSquaredError

    # Arguments parsing
    args = correct_and_smooth_parser()

    # Extraction of masks
    masks = extract_masks(Paths.OBESITY_MASK, k=10, l=0)

    # Genes selection
    if args.genes_subgroup:
        gene_choice = GeneChoice.SIGNIFICANT
    else:
        gene_choice = GeneChoice.ALL

    # Extraction of data
    df, target, cont_cols, cat_cols = get_learning_one_data(PetaleDataManager(), genes=gene_choice)

    # Creation of dataset
    dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False, to_tensor=True)

    # Correct and smooth
    run_correct_and_smooth_experiment(dataset=dataset,
                                      evaluation_name=args.evaluation_name,
                                      masks=masks,
                                      metrics=[AbsoluteError(), Pearson(), SquaredError(), RootMeanSquaredError()],
                                      path=args.path,
                                      r_smooth=args.r_smooth,
                                      r_correct=args.r_correct,
                                      max_degree=args.max_degree)
