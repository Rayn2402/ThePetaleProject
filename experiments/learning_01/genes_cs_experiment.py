"""
This file is used to test logistic regression with correct and smooth
"""

from os.path import dirname, isfile, realpath, join
from os import rename
from tqdm import tqdm
from torch import load, tensor, cat
import sys
import argparse
import time


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 full_experiment.py',
                                     description="Runs all the experiments associated to the l1 dataset")

    # Nb inner split
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=5,
                        help='Number of inner splits during the models evaluations')

    # Complication selection
    parser.add_argument('-comp', '--complication', type=str, default='bone',
                        choices=['bone', 'cardio', 'neuro', 'all'],
                        help='Choices of health complication to predict')

    # Correct and Smooth parameters
    parser.add_argument('--num-correction-layers', type=int, default=50)
    parser.add_argument('--correction-alpha', type=float, default=0.979)
    parser.add_argument('--correction-adj', type=str, default='DAD')
    parser.add_argument('--num-smoothing-layers', type=int, default=50)
    parser.add_argument('--smoothing-alpha', type=float, default=0.756)
    parser.add_argument('--smoothing-adj', type=str, default='DAD')
    parser.add_argument('--scale', type=float, default=20)

    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=SEED, help='Seed to use during model evaluations')

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
    from src.data.extraction.constants import ALL_CHROM_POS
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.sampling import get_learning_one_data, extract_masks, push_valid_to_train
    from src.models.correct_and_smooth import CorrectAndSmooth
    from src.models.mlp import PetaleBinaryMLPC
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.score_metrics import AUC, BinaryAccuracy, BinaryBalancedAccuracy, \
        BalancedAccuracyEntropyRatio, Sensitivity, Specificity, Reduction

    # Arguments parsing
    args = argument_parser()

    # Extraction of complication choice
    complication = args.complication
    if complication == 'bone':
        complication = BONE_COMPLICATIONS
        mask_file = 'l1_bone_mask.json'

    elif complication == 'cardio':
        complication = CARDIOMETABOLIC_COMPLICATIONS
        mask_file = 'l1_cardio_mask.json'

    elif complication == 'neuro':
        complication = NEUROCOGNITIVE_COMPLICATIONS
        mask_file = 'l1_neuro_mask.json'
    else:
        complication = COMPLICATIONS
        mask_file = 'l1_general_mask.json'

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # Extraction of masks
    masks_without_val = extract_masks(join(Paths.MASKS, mask_file), k=1, l=args.nb_inner_splits)
    push_valid_to_train(masks_without_val)
    train_idx, test_idx = masks_without_val[0]['train'], masks_without_val[0]['test']

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [AUC(), BinaryAccuracy(), BinaryBalancedAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(), Specificity(),
                          BalancedAccuracyEntropyRatio(Reduction.GEO_MEAN)]

    # We create a dataset with just baselines
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True, complications=[complication])
    base_dataset = PetaleDataset(df, complication, cont_cols, cat_cols, to_tensor=True)
    base_dataset.update_masks(train_mask=train_idx, test_mask=test_idx)

    # We create another without baselines but with all genes
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=False, complications=[complication], genes='all')
    genes_dataset = PetaleDataset(df, complication, cont_cols, cat_cols, to_tensor=True)
    genes_dataset.update_masks(train_mask=train_idx, test_mask=test_idx)

    # We check if a logistic regression was already trained for the complication
    file_name = f"{args.complication}_logit.pt"
    path = join(Paths.MODELS, file_name)
    if isfile(path):
        print("--Model loading--")
        model = load(path)
    else:
        print("--Model training--")
        model = PetaleBinaryMLPC(n_layer=0, n_unit=2, activation="PReLU", alpha=0.3, beta=0.3, lr=0.02,
                                 batch_size=10, max_epochs=250, num_cont_col=len(base_dataset.cont_cols),
                                 cat_idx=base_dataset.cat_idx, cat_sizes=base_dataset.cat_sizes,
                                 cat_emb_sizes=base_dataset.cat_sizes, weight=0.5)

        # We train the model
        model.fit(base_dataset)

        # We save it and rename the file
        # model.save_model(Paths.MODELS)
        # rename(join(Paths.MODELS, "torch_model.pt"), path)

        # We save the training curve and rename the file
        model.plot_evaluations(Paths.FIGURES_RECORDS)
        rename(join(Paths.FIGURES_RECORDS, "epochs_progression.png"),
               join(Paths.FIGURES_RECORDS, f"{args.complication}_epochs_progression.png"))

    print('--Model evaluation--')
    pred = model.predict_proba(base_dataset)
    model.find_optimal_threshold(base_dataset, evaluation_metrics[-1])
    _, y, _ = base_dataset[test_idx]
    print(f"Threshold : {model.thresh}")
    for m in evaluation_metrics:
        print(f"{m.name} : {m(pred, y, thresh=model.thresh)}")

    print('-- Correction and smoothing with genes--')
    mask_idx = tensor(train_idx)
    _, y, _ = base_dataset[train_idx]
    _, y_test, _ = base_dataset[test_idx]

    # For all chromosome positions
    for chrom_pos in tqdm(ALL_CHROM_POS):

        # We create the correct and smooth object
        cs = CorrectAndSmooth(num_correction_layers=args.num_correction_layers,
                              correction_alpha=args.correction_alpha,
                              correction_adj=args.correction_adj,
                              num_smoothing_layers=args.num_smoothing_layers,
                              smoothing_alpha=args.smoothing_alpha,
                              smoothing_adj=args.smoothing_adj,
                              scale=args.scale)

        # We predict proba on all data
        proba = model.predict_proba(base_dataset, mask=train_idx + test_idx)
        proba = proba.unsqueeze(dim=1)
        proba = cat((1-proba, proba), dim=1)

        # We build the homogeneous graph corresponding to the chromosome connexion
        g = genes_dataset.build_homogeneous_graph(cat_cols=[chrom_pos])

        # We correct and smooth the prediction
        try:
            proba = cs.correct(g, proba, y, mask_idx)
            proba = cs.smooth(g, proba, y, mask_idx)

            # We check the metrics
            print("\n")
            for m in evaluation_metrics:
                print(f"{m.name} : {m(proba[len(train_idx):, 1], y_test)}")

        except RuntimeError:
            print('An error occured')
            pass










