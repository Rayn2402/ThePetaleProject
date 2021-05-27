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
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from Utils.score_metrics import Metric


class Objective(ABC):
    """
    Base class to create objective functions to use with the tuner
    """
    def __init__(self, model_generator: Union[NNModelGenerator, RFCModelGenerator],
                 dataset: Union[PetaleNNDataset, PetaleRFDataset], masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], device: str, metric: Metric, **kwargs):
        """
        Sets protected and public attributes

        Args:
            model_generator: callable object used to generate a model according to a set of hyperparameters
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            device: "cpu" or "gpu"
            metric: callable metric we want to optimize (not used for backpropagation)
        """

        # We set protected attributes
        self._hps = hps
        self._model_generator = model_generator
        self._n_splits = len(masks.keys())
        self._trainer = self._initialize_trainer(dataset, masks, metric, device=device, **kwargs)

    @property
    def metric(self) -> Metric:
        return self._trainer.metric

    @abstractmethod
    def __call__(self, trial: Any) -> float:
        """
        Extracts hyperparameters suggested by optuna and executes "inner_random_subsampling" trainer's function

        Args:
            trial: optuna trial

        Returns: score associated to the set of hyperparameters
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_trainer(self, dataset: Union[PetaleNNDataset, PetaleRFDataset],
                            masks: Dict[int, Dict[str, List[int]]], metric: Metric,
                            **kwargs) -> Union[NNTrainer, RFTrainer]:
        """
        Builds the appropriate trainer object

        Args:
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            metric: callable metric we want to optimize (not used for backpropagation)
            masks: dict with list of idx to use as train, valid and test masks

        Returns: trainer object

        """
        raise NotImplementedError

    @abstractmethod
    def extract_hps(self, trial: Any) -> Dict[str, Any]:
        """
        Given an optuna trial, returns model hyperparameters in a dictionary
        with the appropriate keys

        Args:
            trial: optuna frozen trial

        Returns: dictionary with hyperparameters' values

        """
        raise NotImplementedError


class NNObjective(Objective):
    """
    Neural Networks' objective function to optimize with hyperparameters
    """

    def __init__(self, model_generator: NNModelGenerator,
                 dataset: PetaleNNDataset, masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], device: str, metric: Metric,
                 n_epochs: int, early_stopping: bool = True):
        """
        Sets protected attributes

        Args:
            model_generator: callable object used to generate a model according to a set of hyperparameters
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            device: "cpu" or "gpu"
            metric: callable metric we want to optimize (not used for backpropagation)
            n_epochs: number of epochs
            early_stopping: True if we want to stop the training when the validation loss stops decreasing
        """
        # We make sure that all hyperparameters needed are in hps dict
        for hp in [N_LAYERS, N_UNITS, DROPOUT, BATCH_SIZE, LR, WEIGHT_DECAY, ACTIVATION]:
            assert hp in hps.keys(), f"{hp} is missing from hps dictionary"

        # We call parent's constructor
        super().__init__(model_generator, dataset, masks, hps, device,
                         metric, n_epochs=n_epochs, early_stopping=early_stopping)

        # We set protected methods to extract hyperparameter suggestions
        self._get_activation = self._define_categorical_hp_getter(ACTIVATION)
        self._get_batch_size = self._define_numerical_hp_getter(BATCH_SIZE, INT)
        self._get_dropout = self._define_numerical_hp_getter(DROPOUT, UNIFORM)
        self._get_layers = self._define_layers_getter()
        self._get_lr = self._define_numerical_hp_getter(LR, UNIFORM)
        self._get_weight_decay = self._define_numerical_hp_getter(WEIGHT_DECAY, UNIFORM)

    def __call__(self, trial: Any) -> float:
        """
        Extracts hyperparameters suggested by optuna and executes "inner_random_subsampling" trainer's function

        Args:
            trial: optuna trial

        Returns: score associated to the set of hyperparameters
        """

        # We pick a number of layers and the number of nodes in each hidden layer
        layers = self._get_layers(trial)

        # We pick the dropout probability
        p = self._get_dropout(trial)

        # We pick a value for the batch size
        batch_size = self._get_batch_size(trial)

        # We pick a value for the learning rate
        lr = self._get_lr(trial)

        # We pick a value for the weight decay
        weight_decay = self._get_weight_decay(trial)

        # We pick a type of activation function for the network
        activation = self._get_activation(trial)

        # We define the model with the suggested set of hyper parameters
        model = self._model_generator(layers=layers, dropout=p, activation=activation)

        # We update the Trainer to train our model
        self._trainer.update_trainer(model=model, weight_decay=weight_decay,
                                     batch_size=batch_size, lr=lr, trial=trial)

        # We perform a k fold cross validation to evaluate the model
        score = self._trainer.inner_random_subsampling(self._n_splits)

        # We return the score
        return score

    def _define_categorical_hp_getter(self, hp: str) -> Callable:
        """
        Builds function to properly extract categorical hyperparameters suggestions

        Args:
            hp: name of an hyperparameter

        Returns: function
        """
        if VALUE in self._hps[hp].keys():
            def getter(trial: Any) -> str:
                return self._hps[hp][VALUE]
        else:
            def getter(trial: Any) -> str:
                return trial.suggest_categorical(hp, self._hps[hp][VALUES])

        return getter

    def _define_numerical_hp_getter(self, hp: str, suggest_function: str) -> Callable:
        """
        Builds function to properly extract numerical hyperparameters suggestions

        Args:
            hp: name of an hyperparameter
            suggest_function: optuna suggest function

        Returns: function
        """
        if VALUE in self._hps[hp].keys():
            def getter(trial: Any) -> Union[float, int]:
                return self._hps[hp][VALUE]
        else:
            if suggest_function == INT:
                def getter(trial: Any) -> Union[int]:
                    return trial.suggest_int(hp, self._hps[hp][MIN], self._hps[hp][MAX])
            else:
                def getter(trial: Any) -> Union[float]:
                    return trial.suggest_uniform(hp, self._hps[hp][MIN], self._hps[hp][MAX])

        return getter

    def _define_layers_getter(self) -> Callable:
        """
        Builds function to properly extract layers composition suggestion

        Returns: function
        """
        get_n_layers = self._define_numerical_hp_getter(N_LAYERS, INT)
        n_units = self._hps[N_UNITS].get(VALUE, None)

        if n_units is not None:
            def getter(trial: Any) -> List[int]:
                n_layers = get_n_layers(trial)
                return [n_units]*n_layers
        else:
            def getter(trial: Any) -> List[int]:
                n_layers = get_n_layers(trial)
                return [trial.suggest_int(f"{N_UNITS}{i}", self._hps[N_UNITS][MIN],
                                          self._hps[N_UNITS][MAX]) for i in range(n_layers)]
        return getter

    def _initialize_trainer(self, dataset: PetaleNNDataset,
                            masks: Dict[int, Dict[str, List[int]]],
                            metric: Metric, **kwargs) -> NNTrainer:
        """
        Initializes an NNTrainer object

        Args:
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            metric: callable metric we want to optimize (not used for backpropagation)

        Returns: trainer object

        """
        # Trainer's initialization
        trainer = NNTrainer(model=None, batch_size=None, lr=None, epochs=kwargs['n_epochs'],
                            weight_decay=None, metric=metric, trial=None,
                            early_stopping=kwargs['early_stopping'],
                            device=kwargs['device'])

        # Trainer's parallel process definition
        trainer.define_subprocess(dataset, masks)

        return trainer


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


class Tuner:
    """
    Base of all objects used for hyperparameter tuning
    """
    def __init__(self, study_name: str, objective: Union[NNObjective, RFObjective], n_trials: int,
                 save_hps_importance: Optional[bool] = False,
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
        self.save_hps_importance = save_hps_importance
        self.save_parallel_coordinates = save_parallel_coordinates
        self.save_optimization_history = save_optimization_history

    def get_best_hps(self) -> Dict[str, Any]:
        """
        Retrieves the best hyperparameters found in the tuning
        """
        return self._objective.extract_hps(self._study.best_trial)

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

        if self.save_hps_importance:
            self.plot_hps_importance_graph()

        if self.save_parallel_coordinates:
            self.plot_parallel_coordinates_graph()

        if self.save_optimization_history:
            self.plot_optimization_history_graph()

        # We extract the best hyperparameters and their importance
        best_hps = self.get_best_hps()
        hps_importance = get_param_importances(self._study,
                                               evaluator=FanovaImportanceEvaluator(seed=HYPER_PARAMS_SEED))

        return best_hps, hps_importance

    def plot_hps_importance_graph(self) -> None:
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


# class NNTuner(Tuner):
#     def __init__(self, study_name, model_generator, datasets, hyper_params, l, n_trials, metric,
#                  direction="minimize", max_epochs=100, get_hyperparameters_importance=False,
#                  get_parallel_coordinate=False, get_optimization_history=False,
#                  early_stopping_activated=False, device="cpu", **kwargs):
#         """
#         Class that will be responsible of tuning Neural Networks
#
#         """
#         super().__init__(study_name=study_name, model_generator=model_generator, datasets=datasets,
#                          hyper_params=hyper_params, l=l, n_trials=n_trials, metric=metric,
#                          objective=NNObjective, max_epochs=max_epochs,
#                          early_stopping_activated=early_stopping_activated, direction=direction,
#                          get_hyperparameters_importance=get_hyperparameters_importance,
#                          get_parallel_coordinate=get_parallel_coordinate,
#                          get_optimization_history=get_optimization_history,
#                          device=device, **kwargs)
#
#     def get_best_hyperparams(self):
#         """
#         Method that returns the values of each hyper parameter
#         """
#
#         # we extract the best trial
#         best_trial = self.study.best_trial
#
#         # We extract the best architecture of the model
#         n_units = [key for key in best_trial.params.keys() if "n_units" in key]
#
#         n_layers = self.hyper_params[N_LAYERS][VALUE] if VALUE in self.hyper_params[N_LAYERS].keys() else\
#             best_trial.params[N_LAYERS]
#
#         if len(n_units) > 0:
#             layers = list(map(lambda n_unit: best_trial.params[n_unit], n_units))
#         else:
#             layers = [self.hyper_params[N_UNITS][VALUE] for i in range(n_layers)]
#
#         # We return the best hyperparameters
#         return {
#             LAYERS: layers,
#             DROPOUT: self.hyper_params[DROPOUT][VALUE] if VALUE in self.hyper_params[DROPOUT].keys() else best_trial.params[DROPOUT],
#             LR: self.hyper_params[LR][VALUE] if VALUE in self.hyper_params[LR].keys() else best_trial.params[LR],
#             BATCH_SIZE: self.hyper_params[BATCH_SIZE][VALUE] if VALUE in self.hyper_params[BATCH_SIZE].keys() else best_trial.params[BATCH_SIZE],
#             WEIGHT_DECAY: self.hyper_params[WEIGHT_DECAY][VALUE] if VALUE in self.hyper_params[WEIGHT_DECAY].keys() else best_trial.params[WEIGHT_DECAY],
#             ACTIVATION: self.hyper_params[ACTIVATION][VALUE] if VALUE in self.hyper_params[ACTIVATION].keys() else best_trial.params[ACTIVATION]
#         }
#
#
# class RFTuner(Tuner):
#     def __init__(self, study_name, model_generator, datasets, hyper_params, l, n_trials, metric,
#                  direction="minimize", get_hyperparameters_importance=False, get_parallel_coordinate=False,
#                  get_optimization_history=False, **kwargs):
#         """
#         Class that will be responsible of tuning Random Forests
#
#         """
#         super().__init__(study_name=study_name, model_generator=model_generator, datasets=datasets,
#                          hyper_params=hyper_params, l=l, n_trials=n_trials, metric=metric,
#                          objective=RFObjective, direction=direction,
#                          get_hyperparameters_importance=get_hyperparameters_importance,
#                          get_intermediate_values=get_parallel_coordinate,
#                          get_optimization_history=get_optimization_history, **kwargs)
#
#     def get_best_hyperparams(self):
#         """
#             Method that returns the values of each hyper parameter
#         """
#
#         # We extract the best trial
#         best_trial = self.study.best_trial
#
#         # We return the best hyperparameters
#         return {
#             N_ESTIMATORS: self.hyper_params[N_ESTIMATORS][VALUE] if VALUE in self.hyper_params[N_ESTIMATORS].keys() else best_trial.params[N_ESTIMATORS],
#             MAX_DEPTH: self.hyper_params[MAX_DEPTH][VALUE] if VALUE in self.hyper_params[MAX_DEPTH].keys() else best_trial.params[MAX_DEPTH],
#             MAX_SAMPLES: self.hyper_params[MAX_SAMPLES][VALUE] if VALUE in self.hyper_params[MAX_SAMPLES].keys() else best_trial.params[MAX_SAMPLES],
#             MAX_FEATURES: self.hyper_params[MAX_FEATURES][VALUE] if VALUE in self.hyper_params[MAX_FEATURES].keys() else best_trial.params[MAX_FEATURES],
#         }
