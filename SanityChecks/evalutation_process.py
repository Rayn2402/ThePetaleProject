"""
This file is used to validate the NNEvaluator and RFEvaluator classes
"""


from Data.Datasets import PetaleNNDataset, PetaleRFDataset
from Data.Sampling import get_learning_one_data, extract_masks
from Evaluation.Evaluator import NNEvaluator, RFEvaluator
from Models.GeneralModels import NNClassifier
from Models.ModelGenerator import NNModelGenerator, RFCModelGenerator
from os import chdir
from os.path import dirname, join
from SanityChecks.tuning_process import NN_HPS, RF_HPS
from SQL.constants import *
from SQL.DataManagement.Utils import PetaleDataManager
from Utils.score_metrics import CrossEntropyLoss, Sensitivity, Accuracy


if __name__ == '__main__':

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We make sure we are in the correct working directory
    chdir(dirname(__file__))

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True,
                                                    complications=[NEUROCOGNITIVE_COMPLICATIONS])
    # Extraction of masks
    masks = extract_masks(join(dirname(dirname(__file__)), "Masks", "L1_masks.json"), k=2, l=10)

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

    # Creation of model generator
    model_generator = RFCModelGenerator()

    # Creation of the evaluator
    rf_evaluator = RFEvaluator(model_generator=model_generator, dataset=rf_dataset, masks=masks,
                               hps=RF_HPS, n_trials=50, optimization_metric=metric,
                               evaluation_metrics=evaluation_metrics, save_optimization_history=True)
    # Evaluation
    rf_evaluator.nested_cross_valid()

