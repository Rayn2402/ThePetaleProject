"""
This files is used to validate the Tuner class with NNObjective and RFObjective
"""

import sys

from os.path import join, dirname, realpath


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from sanity_checks.hps import TAB_HPS
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.sampling import get_learning_one_data, extract_masks, SIGNIFICANT
    from src.models.tabnet import PetaleBinaryTNC
    from src.training.tuning import Tuner, Objective
    from src.utils.score_metrics import BalancedAccuracyEntropyRatio, Reduction

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True, genes=SIGNIFICANT,
                                                    complications=[CARDIOMETABOLIC_COMPLICATIONS])

    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "l1_masks.json"), k=1, l=5)
    inner_masks = masks[0]["inner"]
    print(len(inner_masks.values()))

    """
    Tuning validation TabNet
    """
    # Creation of dataset
    dataset = PetaleDataset(df, CARDIOMETABOLIC_COMPLICATIONS, cont_cols, cat_cols)

    # Saving of fixed params for TabNet
    fixed_params = {'cat_idx': dataset.cat_idx, 'cat_sizes': dataset.cat_sizes,
                    'cat_emb_sizes': dataset.cat_sizes, 'max_epochs': 250,
                    'patience': 50}

    # Creation of objective function
    objective = Objective(dataset, inner_masks, TAB_HPS, fixed_params,
                          metric=BalancedAccuracyEntropyRatio(Reduction.GEO_MEAN),
                          model_constructor=PetaleBinaryTNC)

    # Creation of tuner and tuning
    tuner = Tuner(60, objective=objective, save_optimization_history=True, study_name='test')
    best_hps, hps_importance = tuner.tune(verbose=True)

