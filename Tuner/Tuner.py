"""
Authors : Mehdi Mitiche

Files that contains the logic related to hyper parameters tuning

"""
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.visualization import plot_param_importances, plot_intermediate_values, plot_optimization_history
from optuna.logging import FATAL, set_verbosity
from Training.Training import NNTrainer, RFTrainer
from Hyperparameters.constants import *

import os
from pathlib import Path


class NNObjective:
    def __init__(self, model_generator, datasets, hyper_params, k, metric, max_epochs):
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
                            weight_decay=weight_decay, metric=self.metric, trial=trial)

        # We perform a k fold cross validation to evaluate the model
        score = trainer.cross_valid(datasets=self.datasets, k=self.k)

        # We return the score
        return score


class RFObjective:
    def __init__(self, model_generator, datasets, hyper_params, k, metric, max_epochs):
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
        self.max_epochs = max_epochs

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


class Tuner:
    def __init__(self, study_name, model_generator, datasets, hyper_params, k, n_trials, metric, direction="minimize",
                 plot_feature_importance=False, plot_intermediate_values=False, **kwargs):
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
                :param plot_feature_importance: Bool to tell if we want to plot the feature importance graph
                :param plot_intermediate_values: Bool to tell if we want to plot the intermediate values graph

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

        # We save the inputs that will be used when tuning the hyper parameters
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.datasets = datasets
        self.hyper_params = hyper_params
        self.k = k
        self.metric = metric
        self.plot_feature_importance = plot_feature_importance
        self.plot_intermediate_values = plot_intermediate_values

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
                           k=self.k, metric=self.metric, max_epochs=self.max_epochs),
            self.n_trials, n_jobs=1, show_progress_bar=verbose)

        if self.plot_feature_importance:
            # We plot the feature importance graph
            self.plot_feature_importance_graph()

        if self.plot_intermediate_values:
            # We plot the Intermediate values graph
            self.plot_intermediate_values_graph()

        # self.plot_optimization_history()

        # We return the best hyper parameters
        return self.get_best_hyperparams()

    def plot_feature_importance_graph(self):
        """
        Method to plot the hyper parameters graph and save it in a html file
        """

        # We generate the hyper parameters importance graph with optuna
        fig = plot_param_importances(self.study)

        if not os.path.exists('./FeatureImportance/'):
            Path('./FeatureImportance/').mkdir(parents=True, exist_ok=True)

        # We save the graph in a html file to have an interactive graph
        fig.write_html(f"./FeatureImportance/{self.study.study_name}.html")

    def plot_intermediate_values_graph(self):
        """
        Method to plot the Intermediate values graph and save it in a html file
        """

        # We generate the intermediate values graph with optuna
        fig = plot_intermediate_values(self.study)

        if not os.path.exists('./IntermediateValues/'):
            Path('./IntermediateValues/').mkdir(parents=True, exist_ok=True)

        # We save the graph in a html file to have an interactive graph
        fig.write_html(f"./IntermediateValues/{self.study.study_name}.html")

    def plot_optimization_history(self):

        # We generate the intermediate values graph with optuna
        fig = plot_optimization_history(self.study)

        if not os.path.exists('./OptHist/'):
            Path('./OptHist/').mkdir(parents=True, exist_ok=True)

        # We save the graph in a html file to have an interactive graph
        fig.write_html(f"./OptHist/{self.study.study_name}.html")


class NNTuner(Tuner):
    def __init__(self, study_name, model_generator, datasets, hyper_params, k, n_trials, metric,
                 direction="minimize", max_epochs=100, plot_feature_importance=False,
                 plot_intermediate_values=False, **kwargs):
        """
        Class that will be responsible of tuning Neural Networks

        """
        super().__init__(study_name=study_name, model_generator=model_generator, datasets=datasets,
                         hyper_params=hyper_params, k=k, n_trials=n_trials, metric=metric, direction=direction,
                         plot_feature_importance=plot_feature_importance,
                         plot_intermediate_values=plot_intermediate_values, **kwargs)

        self.Objective = NNObjective
        self.max_epochs = max_epochs

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
                 direction="minimize", plot_feature_importance=False,
                 plot_intermediate_values=False, **kwargs):
        """
        Class that will be responsible of tuning Random Forests

        """
        super().__init__(study_name=study_name, model_generator=model_generator, datasets=datasets,
                         hyper_params=hyper_params, k=k, n_trials=n_trials, metric=metric, direction=direction,
                         plot_feature_importance=plot_feature_importance,
                         plot_intermediate_values=plot_intermediate_values, **kwargs)
        self.Objective = RFObjective
        self.max_epochs = None

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
