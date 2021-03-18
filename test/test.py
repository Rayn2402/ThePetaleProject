from SQL.DataManager.Utils import PetaleDataManager
from Models.GeneralModels import NNRegressor, NNClassifier
from Models.ModelGenerator import ModelGenerator
from Utils.score_metrics import ClassificationMetrics
from Datasets.Sampling import LearningOneSampler
from torch import unique, argmax
from Evaluator.NNEvaluator import NNEvaluator

import json

with open("../hyper_params/hyper_params.json", "r") as read_file:
    HYPER_PARAMS = json.load(read_file)

manager = PetaleDataManager("mitm2902")

sampler = LearningOneSampler(dm=manager)
all_data = sampler(k=1, l=1)


# TUNNING NN CLASSIFIER LEARNING 01

def metric01(pred, target):
    return ClassificationMetrics.accuracy(argmax(pred, dim=1).float(), target).item()

cat_sizes = []
for i in range(all_data[0]["train"].X_cat.shape[1]):
    cat_sizes.append(len(unique(all_data[0]["train"].X_cat[:, i])))

generator = ModelGenerator(NNClassifier, num_cont_col=all_data[0]["train"].X_cont.shape[1], cat_sizes=cat_sizes,
                           output_size=3)

evaluator = NNEvaluator(model_generator=generator, sampler=sampler, k=5, l=10, hyper_params=HYPER_PARAMS, n_trials=100,
                        metric=metric01, direction="maximize", seed=2021)
scores = evaluator.nested_cross_valid()
print(scores)

