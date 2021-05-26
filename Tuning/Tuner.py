"""
Authors : Mehdi Mitiche

Files that contains the logic related to hyper parameters tuning

"""

from abc import ABC, abstractmethod
from Data.Datasets import PetaleNNDataset, PetaleRFDataset
from Hyperparameters.constants import *
from Models.ModelGenerator import NNModelGenerator, RFCModelGenerator
from optuna import create_study
from optuna.importance import get_param_importances, FanovaImportanceEvaluator
from optuna.logging import FATAL, set_verbosity
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_param_importances, plot_parallel_coordinate, plot_optimization_history
from os.path import join
from time import strftime
from Training.Trainer import NNTrainer, RFTrainer
from typing import Any, Dict, Optional, Union, Tuple
from Utils.score_metrics import Metric


class NNObjective:
    def __init__(self, model_generator, datasets, hyper_params, l, metric,
                 max_epochs, device, early_stopping_activated=False):
        """
        Class that will represent the objective function for tuning Neural networks
        
        :param model_generator: Instance of the ModelGenerator class that will be responsible of generating the model
        :param datasets: Datasets representing all the train and test sets to be used in the cross validation
        :param hyper_params: Dictionary containing information of the hyper parameter we want to tune
        :param l: Number of folds to use in the internal random subsampling
        :param metric: Function that takes the output of the model and the target
                       and returns the metric we want to optimize
        :param max_epochs: the maximum number of epochs to do in training


        :return: the value of the metric after performing a k-fold cross validation
                 on the model with a subset of the given hyper parameter
        """

        # we save the inputs that will be used when calling the class
        self.model_generator = model_generator
        self.hyper_params = hyper_params
        self.l = l
        self.trainer = NNTrainer(model=None, batch_size=None, lr=None, epochs=max_epochs,
                                 weight_decay=None, metric=metric, trial=None,
                                 early_stopping_activated=early_stopping_activated,
                                 device=device)
        self.trainer.define_subprocess(datasets)

    def __call__(self, trial):
        hyper_params = self.hyper_params

        # We choose a number of hidden layers
        n_layers = hyper_params[N_LAYERS][VALUE] if VALUE in hyper_params[N_LAYERS].keys() else \
            trial.suggest_int(N_LAYERS, hyper_params[N_LAYERS][MIN], hyper_params[N_LAYERS][MAX])

        # We choose a number of node in each hidden layer
        layers = [hyper_params[N_UNITS][VALUE] if VALUE in hyper_params[N_UNITS].keys() else
            trial.suggest_int(f"{N_UNITS}{i}", hyper_params[N_UNITS][MIN], hyper_params[N_UNITS][MAX])
            for i in range(n_layers)]

        # We choose the dropout probability
        p = hyper_params[DROPOUT][VALUE] if VALUE in hyper_params[DROPOUT].keys() else \
            trial.suggest_uniform(DROPOUT, hyper_params[DROPOUT][MIN],hyper_params[DROPOUT][MAX])

        # We choose a value for the batch size used in the training
        batch_size = hyper_params[BATCH_SIZE][VALUE] if VALUE in hyper_params[BATCH_SIZE].keys() else \
            trial.suggest_int(BATCH_SIZE, hyper_params[BATCH_SIZE][MIN], hyper_params[BATCH_SIZE][MAX])

        # We choose a value for the learning rate
        lr = hyper_params[LR][VALUE] if VALUE in hyper_params[LR].keys() else\
            trial.suggest_uniform(LR, hyper_params[LR][MIN], hyper_params[LR][MAX])

        # We choose a value for the weight decay used in the training
        weight_decay = hyper_params[WEIGHT_DECAY][VALUE] if VALUE in hyper_params[WEIGHT_DECAY].keys() else\
            trial.suggest_uniform(WEIGHT_DECAY, hyper_params[WEIGHT_DECAY][MIN], hyper_params[WEIGHT_DECAY][MAX])

        # We choose a type of activation function for the network
        activation = hyper_params[ACTIVATION][VALUE] if VALUE in hyper_params[ACTIVATION].keys() else \
            trial.suggest_categorical(ACTIVATION, hyper_params[ACTIVATION][VALUES])

        # We define the model with the suggested set of hyper parameters
        model = self.model_generator(layers=layers, dropout=p, activation=activation)

        # We update the Trainer to train our model
        self.trainer.update_trainer(model=model, weight_decay=weight_decay, batch_size=batch_size,
                                    lr=lr, trial=trial)

        # We perform a k fold cross validation to evaluate the model
        score = self.trainer.inner_random_subsampling(l=self.l)

        # We return the score
        return score


class RFObjective:
    def __init__(self, model_generator, datasets, hyper_params, l, metric, **kwargs):
        """
        Class that will represent the objective function for tuning Random Forests


        :param model_generator: instance of the ModelGenerator class that will be responsible of generating the model
        :param datasets: Datasets representing all the train and test sets to be used in the cross validation
        :param hyper_params: Dictionary containing information of the hyper parameter we want to tune
        :param l: Number of folds to use in the internal random subsampling
        :param metric: Function that takes the output of the model and the target and returns the metric we want
        to optimize

        :return: Value of the metric after performing a k fold cross validation on the model with a subset of the
        given hyper parameter
        """

        # We save the inputs that will be used when calling the class
        self.model_generator = model_generator
        self.hyper_params = hyper_params
        self.l = l
        self.trainer = RFTrainer(model=None, metric=metric)
        self.trainer.define_subprocess(datasets)

    def __call__(self, trial):
        hyper_params = self.hyper_params

        # We choose the number of estimators used un the training
        n_estimators = hyper_params[N_ESTIMATORS][VALUE] if VALUE in hyper_params[N_ESTIMATORS].keys() else \
            trial.suggest_int(N_ESTIMATORS, hyper_params[N_ESTIMATORS][MIN], hyper_params[N_ESTIMATORS][MAX])

        # We choose the maximum depth of the trees
        max_depth = hyper_params[MAX_DEPTH][VALUE] if VALUE in hyper_params[MAX_DEPTH].keys() else\
            trial.suggest_int(MAX_DEPTH, hyper_params[MAX_DEPTH][MIN], hyper_params[MAX_DEPTH][MAX])

        # We choose a value for the max features to consider in each split
        max_features = hyper_params[MAX_FEATURES][VALUE] if VALUE in hyper_params[MAX_FEATURES].keys() else \
            trial.suggest_uniform(MAX_FEATURES, hyper_params[MAX_FEATURES][MIN], hyper_params[MAX_FEATURES][MAX])

        # We choose a value for the max samples to train for each tree
        max_samples = hyper_params[MAX_SAMPLES][VALUE] if VALUE in hyper_params[MAX_SAMPLES].keys() else\
            trial.suggest_uniform(MAX_SAMPLES, hyper_params[MAX_SAMPLES][MIN], hyper_params[MAX_SAMPLES][MAX])

        # We define the model with the suggested set of hyper parameters
        model = self.model_generator(n_estimators=n_estimators, max_features=max_features,
                                     max_depth=max_depth, max_samples=max_samples)

        # We create the trainer that will train our model
        self.trainer.update_trainer(model=model)

        # We perform a cross validation to evaluate the model
        score = self.trainer.inner_random_subsampling(l=self.l)

        # We return the score
        return score


class Tuner(ABC):
    """
    Base of all objects used for hyperparameter tuning
    """
    def __init__(self, study_name: str, objective: Union[NNObjective, RFObjective], n_trials: int,
                 save_hp_importance: Optional[bool] = False,
                 save_parallel_coordinates: Optional[bool] = False,
                 save_optimization_history: Optional[bool] = False,
                 path: Optional[str] = None, **kwargs):
        """
        Sets all protected and public attributes

        Args:
            study_name: name of the optuna study
            n_trials: number of sets of hyperparameters tested
            objective: objective function to optimize
            save_hp_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
            path: path of the directory used to store graphs created
        """
        # We call super init since we're using ABC
        super().__init__()

        # We set protected attributes
        self._objective = objective
        self._study = create_study(direction=self._objective.metric.direction,
                                   study_name=study_name,
                                   sampler=TPESampler(n_startup_trials=kwargs.get('n_startup_trials', 10),
                                                      n_ei_candidates=kwargs.get('n_ei_candidates', 20),
                                                      multivariate=True),
                                   pruner=NopPruner())

        # We set public attributes
        self.n_trials = n_trials
        self.path = path if path is not None else join("tuning_records", f"{strftime('%Y%m%d-%H%M%S')}")
        self.save_hp_importance = save_hp_importance
        self.save_parallel_coordinates = save_parallel_coordinates
        self.save_optimization_history = save_optimization_history

    def tune(self, verbose: bool = True) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Searches for the hyperparameters that optimize the objective function, using the TPE algorithm

        Args:
            verbose: True if we want optuna to show a progress bar

        Returns: best hyperarameters and hyperparameters' importance

        """

        # We perform the optimization
        set_verbosity(FATAL)  # We remove verbosity from loading bar
        self._study.optimize(self._objective, self.n_trials, show_progress_bar=verbose)

        if self.save_hp_importance:
            self.plot_hp_importance_graph()

        if self.save_parallel_coordinates:
            self.plot_parallel_coordinates_graph()

        if self.save_optimization_history:
            self.plot_optimization_history_graph()

        # We extract the best hyperparameters and their importance
        best_hps = self.get_best_hyperparams()
        hps_importance = get_param_importances(self._study,
                                               evaluator=FanovaImportanceEvaluator(seed=HYPER_PARAMS_SEED))

        return best_hps, hps_importance

    def plot_hp_importance_graph(self) -> None:
        """
        Plots the hyperparameters importance graph and save it in an html file

        Returns: None
        """
        # We generate the hyperparameters importance graph with optuna
        fig = plot_param_importances(self._study, evaluator=FanovaImportanceEvaluator(seed=HYPER_PARAMS_SEED))

        # We save the graph in a html file to have an interactive graph
        fig.write_image(join(self.path, "hp_importance.png"))

    def plot_parallel_coordinates_graph(self) -> None:
        """
        Plots the parallel coordinates graph and save it in an html file
        """

        # We generate the parallel coordinate graph with optuna
        fig = plot_parallel_coordinate(self._study)

        # We save the graph in a html file to have an interactive graph
        fig.write_image(join(self.path, "parallel_coordinates.png"))

    def plot_optimization_history_graph(self) -> None:
        """
        Plots the optimization history graph and save it in a html file
        """

        # We generate the optimization history graph with optuna
        fig = plot_optimization_history(self._study)

        # We save the graph in a html file to have an interactive graph
        fig.write_image(join(self.path, "optimization_history.png"))

    @abstractmethod
    def get_best_hyperparams(self) -> Dict[str, Any]:
        """
        Abstract method to retrieve best hyperparameters
        """
        raise NotImplementedError



class NNTuner(Tuner):
    def __init__(self, study_name, model_generator, datasets, hyper_params, l, n_trials, metric,
                 direction="minimize", max_epochs=100, get_hyperparameters_importance=False,
                 get_parallel_coordinate=False, get_optimization_history=False,
                 early_stopping_activated=False, device="cpu", **kwargs):
        """
        Class that will be responsible of tuning Neural Networks

        """
        super().__init__(study_name=study_name, model_generator=model_generator, datasets=datasets,
                         hyper_params=hyper_params, l=l, n_trials=n_trials, metric=metric,
                         objective=NNObjective, max_epochs=max_epochs,
                         early_stopping_activated=early_stopping_activated, direction=direction,
                         get_hyperparameters_importance=get_hyperparameters_importance,
                         get_parallel_coordinate=get_parallel_coordinate,
                         get_optimization_history=get_optimization_history,
                         device=device, **kwargs)

    def get_best_hyperparams(self):
        """
        Method that returns the values of each hyper parameter
        """

        # we extract the best trial
        best_trial = self.study.best_trial

        # We extract the best architecture of the model
        n_units = [key for key in best_trial.params.keys() if "n_units" in key]

        n_layers = self.hyper_params[N_LAYERS][VALUE] if VALUE in self.hyper_params[N_LAYERS].keys() else\
            best_trial.params[N_LAYERS]

        if len(n_units) > 0:
            layers = list(map(lambda n_unit: best_trial.params[n_unit], n_units))
        else:
            layers = [self.hyper_params[N_UNITS][VALUE] for i in range(n_layers)]

        # We return the best hyperparameters
        return {
            LAYERS: layers,
            DROPOUT: self.hyper_params[DROPOUT][VALUE] if VALUE in self.hyper_params[DROPOUT].keys() else best_trial.params[DROPOUT],
            LR: self.hyper_params[LR][VALUE] if VALUE in self.hyper_params[LR].keys() else best_trial.params[LR],
            BATCH_SIZE: self.hyper_params[BATCH_SIZE][VALUE] if VALUE in self.hyper_params[BATCH_SIZE].keys() else best_trial.params[BATCH_SIZE],
            WEIGHT_DECAY: self.hyper_params[WEIGHT_DECAY][VALUE] if VALUE in self.hyper_params[WEIGHT_DECAY].keys() else best_trial.params[WEIGHT_DECAY],
            ACTIVATION: self.hyper_params[ACTIVATION][VALUE] if VALUE in self.hyper_params[ACTIVATION].keys() else best_trial.params[ACTIVATION]
        }


class RFTuner(Tuner):
    def __init__(self, study_name, model_generator, datasets, hyper_params, l, n_trials, metric,
                 direction="minimize", get_hyperparameters_importance=False, get_parallel_coordinate=False,
                 get_optimization_history=False, **kwargs):
        """
        Class that will be responsible of tuning Random Forests

        """
        super().__init__(study_name=study_name, model_generator=model_generator, datasets=datasets,
                         hyper_params=hyper_params, l=l, n_trials=n_trials, metric=metric,
                         objective=RFObjective, direction=direction,
                         get_hyperparameters_importance=get_hyperparameters_importance,
                         get_intermediate_values=get_parallel_coordinate,
                         get_optimization_history=get_optimization_history, **kwargs)

    def get_best_hyperparams(self):
        """
            Method that returns the values of each hyper parameter
        """

        # We extract the best trial
        best_trial = self.study.best_trial

        # We return the best hyperparameters
        return {
            N_ESTIMATORS: self.hyper_params[N_ESTIMATORS][VALUE] if VALUE in self.hyper_params[N_ESTIMATORS].keys() else best_trial.params[N_ESTIMATORS],
            MAX_DEPTH: self.hyper_params[MAX_DEPTH][VALUE] if VALUE in self.hyper_params[MAX_DEPTH].keys() else best_trial.params[MAX_DEPTH],
            MAX_SAMPLES: self.hyper_params[MAX_SAMPLES][VALUE] if VALUE in self.hyper_params[MAX_SAMPLES].keys() else best_trial.params[MAX_SAMPLES],
            MAX_FEATURES: self.hyper_params[MAX_FEATURES][VALUE] if VALUE in self.hyper_params[MAX_FEATURES].keys() else best_trial.params[MAX_FEATURES],
        }
