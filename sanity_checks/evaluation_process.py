"""
This file is used to validate the NNEvaluator and RFEvaluator classes
"""


import sys
from os.path import join, realpath, dirname


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from sanity_checks.hps import NN_HPS, RF_HPS
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleNNDataset, PetaleRFDataset
    from src.data.processing.sampling import get_learning_one_data, extract_masks
    from src.training.evaluation import NNEvaluator, RFEvaluator
    from src.models.nn_models import NNClassifier
    from src.models.models_generation import NNModelGenerator
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.score_metrics import CrossEntropyLoss, Sensitivity, Accuracy

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True,
                                                    complications=[NEUROCOGNITIVE_COMPLICATIONS])
    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "l1_masks.json"), k=2, l=10)

    # Initialization of the optimization metric
    metric = CrossEntropyLoss()

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = {"Sensitivity": Sensitivity(nb_classes=2), "Accuracy": Accuracy()}

    """
    NNEvaluator validation
    """
    # Creation of dataset
    nn_dataset = PetaleNNDataset(df, NEUROCOGNITIVE_COMPLICATIONS, cont_cols, cat_cols)

    # Creation of model generator
    nb_cont_cols = len(cont_cols)
    cat_sizes = [len(v.items()) for v in nn_dataset.encodings.values()]
    model_generator = NNModelGenerator(NNClassifier, nb_cont_cols, cat_sizes, output_size=2)

    # Creation of the evaluator
    nn_evaluator = NNEvaluator(model_generator=model_generator, dataset=nn_dataset, masks=masks,
                               hps=NN_HPS, n_trials=50, optimization_metric=metric,
                               evaluation_metrics=evaluation_metrics, max_epochs=100, early_stopping=True,
                               save_optimization_history=True)

    # Evaluation
    nn_evaluator.nested_cross_valid()

    """
    RFEvaluator validation
    """
    # Creation of dataset
    rf_dataset = PetaleRFDataset(df, NEUROCOGNITIVE_COMPLICATIONS, cont_cols, cat_cols)

    # Creation of the evaluator
    rf_evaluator = RFEvaluator(dataset=rf_dataset, masks=masks,
                               hps=RF_HPS, n_trials=50, optimization_metric=metric,
                               evaluation_metrics=evaluation_metrics, save_optimization_history=True)
    # Evaluation
    rf_evaluator.nested_cross_valid()

