from SQL.DataManager.Utils import PetaleDataManager
from Models.GeneralModels import NNClassifier
from Models.ModelGenerator import NNModelGenerator
from Evaluator.Evaluator import NNEvaluator
from Utils.score_metrics import ClassificationMetrics as CM
from SQL.NewTablesScripts.constants import SEED
from Datasets.Sampling import LearningOneSampler
from torch import unique
import os
import json

if __name__ == '__main__':

    # HYPERPARAMETER OPTIMIZATION TEST WITH LEARNING 01

    print(f"\nHyperparameter Optimization test...\n")

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")
    sampler = LearningOneSampler(dm=manager)

    # Loading of data
    all_data = sampler(k=1, l=1)
    train = all_data[0]["train"]
    valid = all_data[0]["valid"]

    cat, cont = train.X_cat.shape[1], train.X_cont.shape[1]
    cat_sizes = [len(unique(train.X_cat[:, i])) for i in range(cat)]

    with open(os.path.join("../..", "Hyperparameters", "hyper_params.json"), "r") as read_file:
        HYPER_PARAMS = json.load(read_file)

    generator = NNModelGenerator(NNClassifier, num_cont_col=cont,
                                 cat_sizes=cat_sizes, output_size=3)

    evaluator = NNEvaluator('test', generator, sampler, HYPER_PARAMS,
                            n_trials=200, seed=SEED,  metric=CM.sensitivity_cross,
                            k=3, max_epochs=50, direction="maximize",
                            get_hyperparameters_importance=True,
                            get_parallel_coordinate=True,
                            get_optimization_history=True)

    scores = evaluator.nested_cross_valid(n_startup_trials=10, min_resource=25, eta=2)

    print(scores)
