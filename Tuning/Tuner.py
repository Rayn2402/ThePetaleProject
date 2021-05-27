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
from optuna.study import Study
from optuna.trial import Trial, FrozenTrial
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
                 hps: Dict[str, Dict[str, Any]], device: str, metric: Metric, needed_hps: List[str], **kwargs):
        """
        Sets protected and public attributes

        Args:
            model_generator: callable object used to generate a model according to a set of hyperparameters
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            device: "cpu" or "gpu"
            metric: callable metric we want to optimize (not used for backpropagation)
            needed_hps: list of hyperparameters that needs to be in the hps dictionary
        """
        # We validate the given hyperparameters
        for hp in needed_hps:
            assert hp in hps.keys(), f"'{hp}' is missing from hps dictionary"

        # We set protected attributes
        self._hps = hps
        self._model_generator = model_generator
        self._n_splits = len(masks.keys())
        self._trainer = self._initialize_trainer(dataset, masks, metric, device=device, **kwargs)

    @property
    def metric(self) -> Metric:
        return self._trainer.metric

    def _define_categorical_hp_getter(self, hp: str) -> Callable:
        """
        Builds function to properly extract categorical hyperparameters suggestions

        Args:
            hp: name of an hyperparameter

        Returns: function
        """
        if VALUE in self._hps[hp].keys():
            def getter(trial: Trial) -> str:
                return self._hps[hp][VALUE]
        else:
            def getter(trial: Trial) -> str:
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
            def getter(trial: Trial) -> Union[float, int]:
                return self._hps[hp][VALUE]
        else:
            if suggest_function == INT:
                def getter(trial: Trial) -> Union[int]:
                    return trial.suggest_int(hp, self._hps[hp][MIN], self._hps[hp][MAX])
            else:
                def getter(trial: Trial) -> Union[float]:
                    return trial.suggest_uniform(hp, self._hps[hp][MIN], self._hps[hp][MAX])

        return getter

    @abstractmethod
    def __call__(self, trial: Trial) -> float:
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
    def extract_hps(self, trial: FrozenTrial) -> Dict[str, Any]:
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
    Neural Networks' objective function used to optimize hyperparameters
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

        # We call parent's constructor
        super().__init__(model_generator, dataset, masks, hps, device,
                         metric, n_epochs=n_epochs, early_stopping=early_stopping,
                         needed_hps=[N_LAYERS, N_UNITS, DROPOUT, BATCH_SIZE, LR, WEIGHT_DECAY, ACTIVATION])

        # We set protected methods to extract hyperparameters' suggestions
        self._get_activation = self._define_categorical_hp_getter(ACTIVATION)
        self._get_batch_size = self._define_numerical_hp_getter(BATCH_SIZE, INT)
        self._get_dropout = self._define_numerical_hp_getter(DROPOUT, UNIFORM)
        self._get_layers = self._define_layers_getter()
        self._get_lr = self._define_numerical_hp_getter(LR, UNIFORM)
        self._get_weight_decay = self._define_numerical_hp_getter(WEIGHT_DECAY, UNIFORM)

    def __call__(self, trial: Trial) -> float:
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

    def _define_layers_getter(self) -> Callable:
        """
        Builds function to properly extract layers composition suggestion

        Returns: function
        """
        get_n_layers = self._define_numerical_hp_getter(N_LAYERS, INT)
        n_units = self._hps[N_UNITS].get(VALUE, None)

        if n_units is not None:
            def getter(trial: Trial) -> List[int]:
                n_layers = get_n_layers(trial)
                return [n_units]*n_layers
        else:
            def getter(trial: Trial) -> List[int]:
                n_layers = get_n_layers(trial)
                return [trial.suggest_int(f"{N_UNITS}{i}", self._hps[N_UNITS][MIN],
                                          self._hps[N_UNITS][MAX]) for i in range(n_layers)]
        return getter

    def extract_hps(self, trial: FrozenTrial) -> Dict[str, Any]:
        """
        Given an optuna trial, returns model hyperparameters in a dictionary
        with the appropriate keys

        Args:
            trial: optuna frozen trial

        Returns: dictionary with hyperparameters' values

        """

        # We extract the architecture of the model associated to the frozen trial
        n_units = [key for key in trial.params.keys() if N_UNITS in key]
        n_layers = self._hps[N_LAYERS].get(VALUE, trial.params[N_LAYERS])

        if len(n_units) > 0:
            layers = list(map(lambda n_unit: trial.params[n_unit], n_units))
        else:
            layers = [self._hps[N_UNITS][VALUE]]*n_layers

        # We return the hyperparameters associated to the frozen trial
        return {LAYERS: layers, **{hp: self._hps[hp].get(VALUE, trial.params[hp]) for
                                   hp in [DROPOUT, LR, BATCH_SIZE, WEIGHT_DECAY, ACTIVATION]}}

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
                            weight_decay=None, metric=metric, in_trial=True,
                            early_stopping=kwargs['early_stopping'],
                            device=kwargs['device'])

        # Trainer's parallel process definition
        trainer.define_subprocess(dataset, masks)

        return trainer


class RFObjective(Objective):
    """
    Random Forests' objective function used to optimize hyperparameters
    """
    def __init__(self, model_generator: Union[NNModelGenerator, RFCModelGenerator],
                 dataset: Union[PetaleNNDataset, PetaleRFDataset], masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], device: str, metric: Metric):
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

        # We call parent's constructor
        super().__init__(model_generator, dataset, masks, hps, device, metric,
                         needed_hps=[N_ESTIMATORS, MAX_DEPTH, MAX_FEATURES, MAX_SAMPLES])

        # We set protected methods to extract hyperparameters' suggestions
        self._get_max_depth = self._define_numerical_hp_getter(MAX_DEPTH, INT)
        self._get_max_features = self._define_numerical_hp_getter(MAX_FEATURES, UNIFORM)
        self._get_max_samples = self._define_numerical_hp_getter(MAX_SAMPLES, UNIFORM)
        self._get_n_estimator = self._define_numerical_hp_getter(N_ESTIMATORS, INT)

    def __call__(self, trial):

        # We pick the number of estimators used in the training
        n_estimators = self._get_n_estimator(trial)

        # We pick a value for the maximum depth of the trees
        max_depth = self._get_max_depth(trial)

        # We pick a value for the max features to consider in each split
        max_features = self._get_max_features(trial)

        # We pick a value for the max samples to train for each tree
        max_samples = self._get_max_samples(trial)

        # We define the model with the suggested set of hyper parameters
        model = self._model_generator(n_estimators=n_estimators, max_features=max_features,
                                      max_depth=max_depth, max_samples=max_samples)

        # We update the trainer that will train our model
        self._trainer.update_trainer(model=model)

        # We perform a cross validation to evaluate the model
        score = self._trainer.inner_random_subsampling(self._n_splits)

        # We return the score
        return score

    def extract_hps(self, trial: FrozenTrial) -> Dict[str, Any]:
        """
        Given an optuna trial, returns model hyperparameters in a dictionary
        with the appropriate keys

        Args:
            trial: optuna frozen trial

        Returns: dictionary with hyperparameters' values

        """
        return {hp: self._hps[hp].get(VALUE, trial.params[hp]) for
                hp in [N_ESTIMATORS, MAX_DEPTH, MAX_SAMPLES, MAX_FEATURES]}

    def _initialize_trainer(self, dataset: PetaleRFDataset,
                            masks: Dict[int, Dict[str, List[int]]], metric: Metric,
                            **kwargs) -> RFTrainer:
        """
        Initializes an RFTrainer object

        Args:
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            metric: callable metric we want to optimize (not used for backpropagation)

        Returns: trainer object

        """
        trainer = RFTrainer(model=None, metric=metric)
        trainer.define_subprocess(dataset, masks)

        return trainer


class Tuner:
    """
    Base of all objects used for hyperparameter tuning
    """
    def __init__(self, n_trials: int, study_name: Optional[str] = None,
                 objective: Optional[Union[NNObjective, RFObjective]] = None,
                 save_hps_importance: Optional[bool] = False,
                 save_parallel_coordinates: Optional[bool] = False,
                 save_optimization_history: Optional[bool] = False,
                 path: Optional[str] = None):
        """
        Sets all protected and public attributes

        Args:
            n_trials: number of sets of hyperparameters tested
            study_name: name of the optuna study
            objective: objective function to optimize
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
            path: path of the directory used to store graphs created
        """
        # We call super init since we're using ABC
        super().__init__()

        # We set protected attributes
        self._objective = objective
        self._study = self._new_study(study_name) if study_name is not None else None

        # We set public attributes
        self.n_trials = n_trials
        self.path = path if path is not None else join("tuning_records", f"{strftime('%Y%m%d-%H%M%S')}")
        self.save_hps_importance = save_hps_importance
        self.save_parallel_coordinates = save_parallel_coordinates
        self.save_optimization_history = save_optimization_history

    def _new_study(self, study_name: str) -> Study:
        """
        Creates a new optuna study

        Args:
            study_name: name of the optuna study

        Returns: study object
        """
        return create_study(direction=self._objective.metric.direction,
                            study_name=study_name,
                            sampler=TPESampler(n_startup_trials=20, n_ei_candidates=20, multivariate=True),
                            pruner=NopPruner())

    def _plot_hps_importance_graph(self) -> None:
        """
        Plots the hyperparameters importance graph and save it in an html file

        Returns: None
        """
        # We generate the hyperparameters importance graph with optuna
        fig = plot_param_importances(self._study, evaluator=FanovaImportanceEvaluator(seed=HYPER_PARAMS_SEED))

        # We save the graph in a html file to have an interactive graph
        fig.write_image(join(self.path, "hp_importance.png"))

    def _plot_parallel_coordinates_graph(self) -> None:
        """
        Plots the parallel coordinates graph and save it in an html file
        """

        # We generate the parallel coordinate graph with optuna
        fig = plot_parallel_coordinate(self._study)

        # We save the graph in a html file to have an interactive graph
        fig.write_image(join(self.path, "parallel_coordinates.png"))

    def _plot_optimization_history_graph(self) -> None:
        """
        Plots the optimization history graph and save it in a html file
        """

        # We generate the optimization history graph with optuna
        fig = plot_optimization_history(self._study)

        # We save the graph in a html file to have an interactive graph
        fig.write_image(join(self.path, "optimization_history.png"))

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
        assert (self._study is not None and self._objective is not None), "study and objective must be defined"

        # We perform the optimization
        set_verbosity(FATAL)  # We remove verbosity from loading bar
        self._study.optimize(self._objective, self.n_trials, show_progress_bar=verbose)

        if self.save_hps_importance:
            self._plot_hps_importance_graph()

        if self.save_parallel_coordinates:
            self._plot_parallel_coordinates_graph()

        if self.save_optimization_history:
            self._plot_optimization_history_graph()

        # We extract the best hyperparameters and their importance
        best_hps = self.get_best_hps()
        hps_importance = get_param_importances(self._study,
                                               evaluator=FanovaImportanceEvaluator(seed=HYPER_PARAMS_SEED))

        return best_hps, hps_importance

    def update_tuner(self, study_name: str, objective: Union[NNObjective, RFObjective]) -> None:
        """
        Sets study and objective protected attributes

        Args:
            study_name: name of the optuna study
            objective: objective function to optimize

        Returns: None

        """
        self._study = self._new_study(study_name)
        self._objective = objective

