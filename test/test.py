from SQL.DataManager.Utils import PetaleDataManager
from Models.GeneralModels import NNRegressor, NNClassifier
from Models.ModelGenerator import ModelGenerator
from Training.Training import Trainer
from Utils.score_metrics import ClassificationMetrics
from Utils.visualization import visualize_epoch_losses
from Datasets.Sampling import LearningOneSampler
from torch import unique, argmax, manual_seed
import numpy as np
import os
from Evaluator.NNEvaluator import NNEvaluator

import json


if __name__ == '__main__':

    # OVERFIT TEST #

    """
    The training loss should move towards 0 while the validation
    loss should decrease at first and then increase
    """
    # We set the seed for the sampling part
    np.random.seed(2021)

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")
    sampler = LearningOneSampler(dm=manager)

    # Loading of data
    all_data = sampler(k=1, l=1)
    train = all_data[0]["train"]
    valid = all_data[0]["valid"]

    cat, cont = train.X_cat.shape[1], train.X_cont.shape[1]
    cat_sizes = [len(unique(train.X_cat[:, i])) for i in range(cat)]

    print(f"\nOverfitting test...\n")

    # We set the seed for the model
    manual_seed(2021)

    # Creation of a simple model
    Model = NNClassifier(num_cont_col=cont, output_size=3, layers=[10, 20, 20],
                         activation='ReLU', cat_sizes=cat_sizes)

    # Creation of a Trainer
    trainer = Trainer(Model)

    # Training for 50 epochs
    t_loss, v_loss = trainer.fit(train, valid, batch_size=20, lr=0.001,
                                 weight_decay=0, epochs=1000, early_stopping_activated=False)

    # Visualization of the losses
    visualize_epoch_losses(t_loss, v_loss)

    # WEIGHT DECAY TEST #

    """
    The training loss should be higher since we are increasing the L2 Penalty
    """
    print(f"\nWeight decay test...\n")
    for decay in [0, 1, 2]:

        # We set the seed for the model
        manual_seed(2021)

        # Creation of a simple model
        Model = NNClassifier(num_cont_col=cont, output_size=3, layers=[10, 20, 20],
                             activation='ReLU', cat_sizes=cat_sizes)

        # Creation of a Trainer
        trainer = Trainer(Model)

        # Training for 100 epochs
        t_loss, v_loss = trainer.fit(train, valid, batch_size=20, lr=0.001,
                                     weight_decay=decay, epochs=100, early_stopping_activated=False)

        # Visualization of the losses
        visualize_epoch_losses(t_loss, v_loss)

    # EARLY STOPPING TEST #
    print(f"\nEarly stopping test...\n")
    # We set the seed for the model
    manual_seed(2021)

    # Creation of a simple model
    Model = NNClassifier(num_cont_col=cont, output_size=3, layers=[10, 20, 20],
                         activation='ReLU', cat_sizes=cat_sizes)

    # Creation of a Trainer
    trainer = Trainer(Model)

    # Training for 1000 epochs
    t_loss, v_loss = trainer.fit(train, valid, batch_size=20, lr=0.001,
                                 weight_decay=decay, epochs=1000, early_stopping_activated=True, patience=15)

    # Visualization of the losses
    visualize_epoch_losses(t_loss, v_loss)

    # HYPERPARAMETER OPTIMIZATION TEST WITH LEARNING 01
    print(f"\nHyperparameter Optimization test...\n")

    with open(os.path.join("..", "Hyperparameters", "hyper_params.json"), "r") as read_file:
        HYPER_PARAMS = json.load(read_file)

    def metric01(pred, target):
        return ClassificationMetrics.accuracy(argmax(pred, dim=1).float(), target).item()

    generator = ModelGenerator(NNClassifier, num_cont_col=cont, cat_sizes=cat_sizes, output_size=3)

    evaluator = NNEvaluator(model_generator=generator, sampler=sampler, k=1, l=1,
                            hyper_params=HYPER_PARAMS, n_trials=100, metric=metric01,
                            direction="maximize", seed=2021, study_name='bob', max_epochs=1000)

    scores = evaluator.nested_cross_valid(min_resource=50)
    print(scores)

