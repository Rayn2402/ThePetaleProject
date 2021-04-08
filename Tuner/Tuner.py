"""
Authors : Mehdi Mitiche

Files that contains the logic related to hyper parameters tuning

"""
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.importance import get_param_importances, FanovaImportanceEvaluator
from optuna.visualization import plot_param_importances, plot_parallel_coordinate, plot_optimization_history
from optuna.logging import FATAL, set_verbosity
from Trainer.Trainer import NNTrainer, RFTrainer
from Hyperparameters.constants import *

import os
from pathlib import Path


class NNObjective:
    def __init__(self, model_generator, datasets, hyper_params, k, metric, max_epochs, early_stopping_activated=False):
        """
        Class that will represent the objective function for tuning Neural networks
        
        :param model_generator: Instance of the ModelGenerator class that will be responsible of generating the model
        :param datasets: Datasets representing all the train and test sets to be used in the cross validation
        :param hyper_params: Dictionary containing information of the hyper parameter we want to tune
        :param k: Number of folds to use in the cross validation
        :param metric: Function that takes the output of the model and the target
                       and returns the metric we want to optimize
        :param max_epochs: the maximum number of epochs to do in training


        :return: the value of the metric after performing a k-fold cross validation
                 on the model with a subset of the given hyper parameter
        """

        # we save the inputs that will be used when calling the class
        self.model_generator = model_generator
        self.datasets = datasets
        self.hyper_params = hyper_params
        self.k = k
        self.metric = metric
        self.max_epochs = max_epochs
        self.early_stopping_activated = early_stopping_activated

    def __call__(self, trial):
        hyper_params = self.hyper_params

        # We choose a number of hidden layers
        n_layers = trial.suggest_int(N_LAYERS,
                                     hyper_params[N_LAYERS][MIN],
                                     hyper_params[N_LAYERS][MAX])

        # We choose a number of node in each hidden layer
        layers = [trial.suggest_int(f"{N_UNITS}{i}", hyper_params[N_UNITS][MIN],
                                    hyper_params[N_UNITS][MAX]) for i in range(n_layers)]

        # We choose the dropout probability
        p = trial.suggest_uniform(DROPOUT,
                                  hyper_params[DROPOUT][MIN],
                                  hyper_params[DROPOUT][MAX])

        # We choose a value for the batch size used in the training
        batch_size = trial.suggest_int(BATCH_SIZE,
                                       hyper_params[BATCH_SIZE][MIN],
                                       hyper_params[BATCH_SIZE][MAX])

        # We choose a value for the learning rate
        lr = trial.suggest_uniform(LR,
                                   hyper_params[LR][MIN],
                                   hyper_params[LR][MAX])

        # We choose a value for the weight decay used in the training
        weight_decay = trial.suggest_uniform(WEIGHT_DECAY,
                                             hyper_params[WEIGHT_DECAY][MIN],
                                             hyper_params[WEIGHT_DECAY][MAX])

        # We choose a type of activation function for the network
        activation = trial.suggest_categorical(ACTIVATION,
                                               hyper_params[ACTIVATION][VALUES])

        # We define the model with the suggested set of hyper parameters
        model = self.model_generator(layers=layers, dropout=p, activation=activation)

        # We create the Trainer that will train our model
        trainer = NNTrainer(model=model, batch_size=batch_size, lr=lr, epochs=self.max_epochs,
                            weight_decay=weight_decay, metric=self.metric, trial=trial,
                            early_stopping_activated=self.early_stopping_activated)

        # We perform a k fold cross validation to evaluate the model
        score = trainer.cross_valid(datasets=self.datasets, k=self.k)

        # We return the score
        return score


class RFObjective:
    def __init__(self, model_generator, datasets, hyper_params, k, metric, **kwargs):
        """
        Class that will represent the objective function for tuning Random Forests


        :param model_generator: instance of the ModelGenerator class that will be responsible of generating the model
        :param datasets: Datasets representing all the train and test sets to be used in the cross validation
        :param hyper_params: Dictionary containing information of the hyper parameter we want to tune
        :param k: Number of folds to use in the cross validation
        :param metric: Function that takes the output of the model and the target and returns the metric we want
        to optimize

        :return: Value of the metric after performing a k fold cross validation on the model with a subset of the
        given hyper parameter
        """

        # We save the inputs that will be used when calling the class
        self.model_generator = model_generator
        self.datasets = datasets
        self.hyper_params = hyper_params
        self.k = k
        self.metric = metric

    def __call__(self, trial):
        hyper_params = self.hyper_params

        # We choose the number of estimators used un the training
        n_estimators = trial.suggest_int(N_ESTIMATORS, hyper_params[N_ESTIMATORS][MIN],
                                         hyper_params[N_ESTIMATORS][MAX])

        # We choose the maximum depth of the trees
        max_depth = trial.suggest_int(MAX_DEPTH, hyper_params[MAX_DEPTH][MIN], hyper_params[MAX_DEPTH][MAX])

        # We choose a value for the max features to consider in each split
        max_features = trial.suggest_uniform(MAX_FEATURES,
                                             hyper_params[MAX_FEATURES][MIN],
                                             hyper_params[MAX_FEATURES][MAX])

        # We choose a value for the max samples to train for each tree
        max_samples = trial.suggest_uniform(MAX_SAMPLES,
                                            hyper_params[MAX_SAMPLES][MIN],
                                            hyper_params[MAX_SAMPLES][MAX])

        # We define the model with the suggested set of hyper parameters
        model = self.model_generator(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                     max_samples=max_samples)

        # We create the trainer that will train our model
        trainer = RFTrainer(model=model, metric=self.metric)

        # We perform a cross validation to evaluate the model
        score = trainer.cross_valid(datasets=self.datasets, k=self.k)

        # We return the score
        return score


HYPER_PARAMS_SEED = 2021


class Tuner:
    def __init__(self, study_name, model_generator, datasets, hyper_params, k, n_trials, metric, direction="minimize",
                 get_hyperparameters_importance=False, get_parallel_coordinate=False, get_optimization_history=False,
                 path=None, **kwargs):
        """
                Class that will be responsible of the hyperparameters tuning

                :param study_name: String that represents the name of the study
                :param model_generator: Instance of the ModelGenerator class that will be responsible of generating
                 the model
                :param datasets: Petale Dataset representing all the train and test sets to be used in the cross
                 validation
                :param hyper_params: Dictionary containing information of the hyper parameter we want to tune
                :param k: Number of folds to use in the cross validation
                :param metric: Function that takes the output of the model and the target and returns
                the metric we want to optimize
                :param n_trials: Number of trials we want to perform
                :param direction: String to specify if we want to maximize or minimize the value of the metric used
                :param get_hyperparameters_importance: Bool to tell if we want to plot the hyperparameters importance
                                                        graph
                :param get_parallel_coordinate: Bool to tell if we want to plot the parallel_coordinate graph
                :param get_optimization_history: Bool to tell if we want to plot the optimization history graph
                :param path: String that represents the path to the folder where we want to save our graphs


                """

        # We look for keyword args
        n_startup = kwargs.get('n_startup_trials', 10)
        n_ei_candidates = kwargs.get('n_ei_candidates', 20)
        min_resource = kwargs.get('min_resource', 10)
        eta = kwargs.get('eta', 4)

        # we create the study 
        self.study = create_study(direction=direction, study_name=study_name,
                                  sampler=TPESampler(n_startup_trials=n_startup,
                                                     n_ei_candidates=n_ei_candidates,
                                                     multivariate=True),
                                  pruner=SuccessiveHalvingPruner(min_resource=min_resource,
                                                                 reduction_factor=eta, bootstrap_count=10))

        assert not ((get_optimization_history or get_parallel_coordinate or get_hyperparameters_importance) and (
                path is None)), "Path to the folder where save graphs must be specified "

        # We save the inputs that will be used when tuning the hyper parameters
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.datasets = datasets
        self.hyper_params = hyper_params
        self.k = k
        self.metric = metric
        self.get_hyperparameters_importance = get_hyperparameters_importance
        self.get_parallel_coordinate = get_parallel_coordinate
        self.get_optimization_history = get_optimization_history
        self.path = path

    def tune(self, verbose=True):
        """
        Method to call to tune the hyperparameters of a given model

        :return: the result of the study containing the best trial and the best values of each hyper parameter
        """

        # We perform the optimization
        set_verbosity(FATAL)  # We remove verbosity from loading bar
        self.study.optimize(
            self.Objective(model_generator=self.model_generator,
                           datasets=self.datasets,
                           hyper_params=self.hyper_params,
                           k=self.k, metric=self.metric, max_epochs=self.max_epochs,
                           early_stopping_activated=self.early_stopping_activated
                           ),
            self.n_trials, n_jobs=1, show_progress_bar=verbose)

        if self.get_hyperparameters_importance:
            # We plot the hyperparameters importance graph
            self.plot_hyperparameters_importance_graph()

        if self.get_parallel_coordinate:
            # We plot the parallel coordinate graph
            self.plot_parallel_coordinate_graph()

        if self.get_optimization_history:
            # We plot the optimization history graph
            self.plot_optimization_history_graph()

        # We return the best hyper parameters and the hyperparameters importance
        return self.get_best_hyperparams(), get_param_importances(study=self.study,
                                                                  evaluator=FanovaImportanceEvaluator(
                                                                      seed=HYPER_PARAMS_SEED))

    def plot_hyperparameters_importance_graph(self):
        """
        Method to plot the hyper parameters graph and save it in a html file
        """

        # We generate the hyper parameters importance graph with optuna
        fig = plot_param_importances(self.study, evaluator=FanovaImportanceEvaluator(
            seed=HYPER_PARAMS_SEED))

        # We save the graph in a html file to have an interactive graph
        fig.write_html(os.path.join(self.path, "hyperparameters_importance.html"))

    def plot_parallel_coordinate_graph(self):
        """
        Method to plot the parallel coordinate graph and save it in a html file
        """

        # We generate the parallel coordinate graph with optuna
        fig = plot_parallel_coordinate(self.study)

        # We save the graph in a html file to have an interactive graph
        fig.write_html(os.path.join(self.path, "parallel_coordinate.html"))

    def plot_optimization_history_graph(self):
        """
        Method to plot the optimization history graph and save it in a html file
        """

        # We generate the optimization history graph with optuna
        fig = plot_optimization_history(self.study)

        # We save the graph in a html file to have an interactive graph
        fig.write_html(os.path.join(self.path, "optimization_history.html"))


class NNTuner(Tuner):
    def __init__(self, study_name, model_generator, datasets, hyper_params, k, n_trials, metric,
                 direction="minimize", max_epochs=100, get_hyperparameters_importance=False,
                 get_parallel_coordinate=False, get_optimization_history=False,
                 early_stopping_activated=False, **kwargs):
        """
        Class that will be responsible of tuning Neural Networks

        """
        super().__init__(study_name=study_name, model_generator=model_generator, datasets=datasets,
                         hyper_params=hyper_params, k=k, n_trials=n_trials, metric=metric, direction=direction,
                         get_hyperparameters_importance=get_hyperparameters_importance,
                         get_parallel_coordinate=get_parallel_coordinate,
                         get_optimization_history=get_optimization_history, **kwargs)
        self.Objective = NNObjective
        self.max_epochs = max_epochs
        self.early_stopping_activated = early_stopping_activated

    def get_best_hyperparams(self):
        """
        Method that returns the values of each hyper parameter
        """

        # we extract the best trial
        best_trial = self.study.best_trial

        # We extract the best architecture of the model
        n_units = [key for key in best_trial.params.keys() if "n_units" in key]
        if n_units is not None:
            layers = list(map(lambda n_unit: best_trial.params[n_unit], n_units))
        else:
            layers = []

        # We return the best hyperparameters
        return {
            LAYERS: layers,
            DROPOUT: best_trial.params[DROPOUT],
            LR: best_trial.params[LR],
            BATCH_SIZE: best_trial.params[BATCH_SIZE],
            WEIGHT_DECAY: best_trial.params[WEIGHT_DECAY],
            ACTIVATION: best_trial.params[ACTIVATION]
        }


class RFTuner(Tuner):
    def __init__(self, study_name, model_generator, datasets, hyper_params, k, n_trials, metric,
                 direction="minimize", get_hyperparameters_importance=False, get_parallel_coordinate=False,
                 get_optimization_history=False, **kwargs):
        """
        Class that will be responsible of tuning Random Forests

        """
        super().__init__(study_name=study_name, model_generator=model_generator, datasets=datasets,
                         hyper_params=hyper_params, k=k, n_trials=n_trials, metric=metric, direction=direction,
                         get_hyperparameters_importance=get_hyperparameters_importance,
                         get_intermediate_values=get_parallel_coordinate,
                         get_optimization_history=get_optimization_history, **kwargs)
        self.Objective = RFObjective
        self.max_epochs = None
        self.early_stopping_activated = None

    def get_best_hyperparams(self):
        """
            Method that returns the values of each hyper parameter
        """

        # We extract the best trial
        best_trial = self.study.best_trial

        # We return the best hyperparameters
        return {
            N_ESTIMATORS: best_trial.params[N_ESTIMATORS],
            MAX_DEPTH: best_trial.params[MAX_DEPTH],
            MAX_SAMPLES: best_trial.params[MAX_SAMPLES],
            MAX_FEATURES: best_trial.params[MAX_FEATURES],

        }
