"""
This files is used to validate the Tuner class with NNObjective and RFObjective
"""

from Data.Datasets import PetaleNNDataset, PetaleRFDataset
from Data.Sampling import get_learning_one_data, extract_masks, SIGNIFICANT
from Hyperparameters.constants import *
from Models.GeneralModels import NNClassifier
from Models.ModelGenerator import NNModelGenerator, RFCModelGenerator
from os import chdir
from os.path import dirname, join
from SQL.constants import *
from SQL.DataManagement.Utils import PetaleDataManager
from Tuning.Tuner import Tuner, NNObjective, RFObjective
from Utils.score_metrics import SensitivityCrossEntropyRatio

NN_HPS = {
    LR: {
        MIN: 1e-2,
        MAX: 1e-1
    },
    BATCH_SIZE: {
        VALUE: 50
    },
    N_LAYERS: {
        MIN: 1,
        MAX: 3,
    },
    N_UNITS: {
        MIN: 2,
        MAX: 5,
    },
    DROPOUT: {
        VALUE: 0
    },
    ACTIVATION: {
        VALUE: "ReLU"
    },
    WEIGHT_DECAY: {
        VALUE: 0.1
    }
}

RF_HPS = {
    N_ESTIMATORS: {
        MIN: 80,
        MAX: 120,
    },
    MAX_FEATURES: {
        MIN: .8,
        MAX: 1,
    },
    MAX_SAMPLES: {
        MIN: .6,
        MAX: .8,
    },
    MAX_DEPTH: {
        VALUE: 50
    }
}

if __name__ == '__main__':

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We make sure we are in the correct working directory
    chdir(dirname(__file__))

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True,
                                                    complications=[CARDIOMETABOLIC_COMPLICATIONS])

    # Extraction of masks
    masks = extract_masks(join(dirname(dirname(__file__)), "Masks", "L1_masks.json"), k=1, l=10)
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