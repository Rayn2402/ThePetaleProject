"""
This files is used to validate the Tuner class with NNObjective and RFObjective
"""

import sys

from os.path import join, dirname, realpath


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from sanity_checks.hps import NN_HPS, RF_HPS
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleRFDataset, PetaleNNDataset
    from src.data.processing.sampling import get_learning_one_data, extract_masks
    from src.models.nn_models import NNClassifier
    from src.models.models_generation import NNModelGenerator, RFCModelGenerator
    from src.training.tuning import Tuner, NNObjective, RFObjective
    from src.utils.score_metrics import SensitivityCrossEntropyRatio

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True,
                                                    complications=[CARDIOMETABOLIC_COMPLICATIONS])

    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "l1_masks.json"), k=1, l=10)
    inner_masks = masks[0]["inner"]
    train_mask, valid_mask, test_mask = masks[0]["train"], masks[0]["valid"], masks[0]["test"]

    """
    Tuning validation with Neural Networks
    """
    # Creation of dataset
    nn_dataset = PetaleNNDataset(df, CARDIOMETABOLIC_COMPLICATIONS, cont_cols, cat_cols)

    # Creation of model generator
    nb_cont_cols = len(cont_cols)
    cat_sizes = [len(v.items()) for v in nn_dataset.encodings.values()]
    model_generator = NNModelGenerator(NNClassifier, nb_cont_cols, cat_sizes, output_size=2)

    # Initialization of a metric
    metric = SensitivityCrossEntropyRatio(nb_classes=2)

    # Creation of NNobjective
    nn_objective = NNObjective(model_generator=model_generator, dataset=nn_dataset,
                               masks=inner_masks, hps=NN_HPS, device="cpu", metric=metric,
                               n_epochs=100, early_stopping=True)

    # Creation of tuner
    tuner = Tuner(n_trials=25, study_name="sanity_check", objective=nn_objective,
                  save_hps_importance=True, save_parallel_coordinates=True, save_optimization_history=True)

    # Tuning
    best_hps, hps_importance = tuner.tune()

    """
    Tuning validation with Random Forest Classifier
    """
    # Creation of dataset
    rf_dataset = PetaleRFDataset(df, CARDIOMETABOLIC_COMPLICATIONS, cont_cols, cat_cols)

    # Creation of model generator
    model_generator = RFCModelGenerator()

    # Creation of RFObjective
    rf_objective = RFObjective(model_generator=model_generator, dataset=rf_dataset, masks=inner_masks,
                               hps=RF_HPS, device="cpu", metric=metric)

    # Creation of tuner
    tuner = Tuner(n_trials=50, study_name="sanity_check_2", objective=rf_objective,
                  save_hps_importance=True, save_parallel_coordinates=True, save_optimization_history=True)

    # Tuning
    best_hps, hps_importance = tuner.tune()
