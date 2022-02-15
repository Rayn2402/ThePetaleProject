"""
Filename: tuning.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to define the Objective and Tuner classes
             used for hyperparameter tuning

Date of last modification : 2021/10/29
"""
import ray

from copy import deepcopy
from optuna import create_study
from optuna.importance import get_param_importances, FanovaImportanceEvaluator
from optuna.logging import FATAL, set_verbosity
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import Trial, FrozenTrial
from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_optimization_history
from os import makedirs
from os.path import join
from settings.paths import Paths
from src.data.processing.datasets import MaskType, PetaleDataset
from src.models.abstract_models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.utils.score_metrics import Direction, Metric
from src.utils.hyperparameters import CategoricalHP, Distribution, HP, NumericalContinuousHP, NumericalIntHP, Range
from time import strftime
from torch import mean, tensor
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


class Objective:
    """
    Callable objective function to use with the tuner
    """
    def __init__(self,
                 dataset: PetaleDataset,
                 masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]],
                 fixed_params: Optional[Dict[str, Any]],
                 metric: Optional[Metric],
                 model_constructor: Callable,
                 gpu_device: bool = False):
        """
        Sets protected and public attributes of the objective

        Args:
            dataset: custom dataset containing all the data needed for our evaluations
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            metric: callable metric we want to optimize with hyperparameters (not used for backpropagation)
            model_constructor: callable object that builds a model using hyperparameters and fixed params
            gpu_device: true if we want to use a gpu
        """
        # We validate the given hyperparameters
        for hp in model_constructor.get_hps():
            if not (hp.name in list(hps.keys())):
                raise ValueError(f"'{hp}' is missing from hps dictionary")

        # We validate the given metric
        if metric is None and not model_constructor.is_encoder():
            raise ValueError('A metric must be specified for this type of model constructor')

        # We set protected attributes
        self._dataset = dataset
        self._fixed_params = fixed_params if fixed_params is not None else {}
        self._hps = hps
        self._masks = masks
        self._metric = metric
        self._model_constructor = model_constructor
        self._getters = {}
        self._define_getters()

        # We set the protected parallel evaluation method
        self._run_single_evaluation = self._build_parallel_process(gpu_device)

    @property
    def metric(self):
        return self._metric

    def __call__(self, trial: Trial) -> float:
        """
        Extracts hyperparameters suggested by optuna and executes
        the parallel evaluations of the hyperparameters set

        Args:
            trial: optuna trial

        Returns: score associated to the set of hyperparameters
        """
        # We extract hyperparameters suggestions
        suggested_hps = {k: f(trial) for k, f in self._getters.items()}

        # We execute parallel evaluations
        futures = [self._run_single_evaluation.remote(masks=m, hps=suggested_hps) for k, m in self._masks.items()]
        scores = ray.get(futures)

        # We take the mean of the scores
        return mean(tensor(scores)).item()

    def _define_getters(self) -> None:
        """
        Defines the different optuna sampling function for each hyperparameter
        """
        # For each hyperparameter associated to the model we are tuning
        for hp in self._model_constructor.get_hps():

            # We check if a value was predefined for this hyperparameter,
            # in this case, no value we'll be sampled by Optuna
            if Range.VALUE in self._hps[hp.name].keys():
                self._getters[hp.name] = self._build_constant_getter(hp)

            # Otherwise we build the suggestion function appropriate to the hyperparameter
            elif hp.distribution == Distribution.CATEGORICAL:
                self._getters[hp.name] = self._build_categorical_getter(hp)

            elif hp.distribution == Distribution.UNIFORM:
                self._getters[hp.name] = self._build_numerical_cont_getter(hp)

            else:  # discrete uniform distribution
                self._getters[hp.name] = self._build_numerical_int_getter(hp)

    def _build_constant_getter(self, hp: HP) -> Callable:
        """
        Builds a function that extracts the given predefined hyperparameter value

        Args:
            hp: hyperparameter

        Returns: function
        """

        def getter(trial: Trial) -> Any:
            return self._hps[hp.name][Range.VALUE]

        return getter

    def _build_categorical_getter(self, hp: CategoricalHP) -> Callable:
        """
        Builds a function that extracts optuna's suggestion for the categorical hyperparameter

        Args:
            hp: categorical hyperparameter

        Returns: function
        """

        def getter(trial: Trial) -> str:
            return trial.suggest_categorical(hp.name, self._hps[hp.name][Range.VALUES])

        return getter

    def _build_numerical_int_getter(self, hp: NumericalIntHP) -> Callable:
        """
        Builds a function that extracts optuna's suggestion for the numerical discrete hyperparameter

        Args:
            hp: numerical discrete hyperparameter

        Returns: function
        """
        def getter(trial: Trial) -> int:
            return trial.suggest_int(hp.name, self._hps[hp.name][Range.MIN], self._hps[hp.name][Range.MAX],
                                     step=self._hps[hp.name].get(Range.STEP, 1))
        return getter

    def _build_numerical_cont_getter(self, hp: NumericalContinuousHP) -> Callable:
        """
        Builds a function that extracts optuna's suggestion for the numerical continuous hyperparameter

        Args:
            hp: numerical continuous hyperparameter

        Returns: function
        """
        def getter(trial: Trial) -> Union[float]:
            return trial.suggest_uniform(hp.name, self._hps[hp.name][Range.MIN], self._hps[hp.name][Range.MAX])

        return getter

    def _build_parallel_process(self, gpu_device: bool = False) -> Callable:
        """
        Builds the function run in parallel for each set of hyperparameters
        and return the score

        Args:
            gpu_device: true indicates that we want to use a gpu

        Returns: function
        """
        @ray.remote(num_gpus=int(gpu_device))
        def run_single_evaluation(masks: Dict[str, List[int]],
                                  hps: Dict[str, Any]) -> float:
            """
            Train a single model using given masks and given hyperparameters

            Args:
                masks: dictionary with list of integers for train, valid and test mask
                hps: dictionary with hyperparameters to give to the model constructor

            Returns: metric score
            """
            # We extract masks
            train_idx, valid_idx, test_idx = masks[MaskType.TRAIN], masks[MaskType.VALID], masks[MaskType.TEST]

            # We create a copy of the current dataset and update its masks
            dts = deepcopy(self._dataset)
            dts.update_masks(train_mask=train_idx, valid_mask=valid_idx, test_mask=test_idx)

            # We build a model using hps and fixed params (PetaleRegressor, PetaleClassifier or PetaleEncoder)
            model = self._model_constructor(**hps, **self._fixed_params)

            # We train the model
            model.fit(dts)

            # If the model is a classifier we find its optimal threshold
            # and compute the prediction score
            if isinstance(model, PetaleBinaryClassifier):
                model.find_optimal_threshold(dataset=dts, metric=self._metric)
                pred = model.predict_proba(dataset=dts)
                _, y, _ = dts[dts.test_mask]
                score = self._metric(pred, y, thresh=model.thresh)

            # If the model is a regression model, we compute the prediction score
            elif isinstance(model, PetaleRegressor):
                pred = model.predict(dataset=dts)
                _, y, _ = dts[dts.test_mask]
                score = self._metric(pred, y)

            # Otherwise, if its an encoder, we calculate the loss
            else:
                pred = model.predict(dataset=dts)
                score = model.loss(pred, dts.test_mask)

            return score

        return run_single_evaluation

    def extract_hps(self, trial: FrozenTrial) -> Dict[str, Any]:
        """
        Given an optuna trial, returns model hyperparameters in a dictionary
        with the appropriate keys

        Args:
            trial: optuna frozen trial

        Returns: dictionary with hyperparameters' values

        """
        return {hp.name: self._hps[hp.name].get(Range.VALUE, trial.params.get(hp.name))
                for hp in self._model_constructor.get_hps()}


class Tuner:
    """
    Object in charge of hyperparameter tuning
    """
    # HYPERPARAMETERS IMPORTANCE SEED
    HP_IMPORTANCE_SEED: int = 2021

    # FIGURES NAME
    HPS_IMPORTANCE_FIG: str = "hp_importance.png"
    PARALLEL_COORD_FIG: str = "parallel_coordinates.png"
    OPTIMIZATION_HIST_FIG: str = "optimization_history.png"

    def __init__(self,
                 n_trials: int,
                 study_name: Optional[str] = None,
                 objective: Objective = None,
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
            save_hps_importance: true if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: true if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: true if we want to plot the optimization history graph after tuning
            path: path of the directory used to store graphs created
        """

        # We set protected attributes
        self._objective = objective
        self._study = self._new_study(study_name) if study_name is not None else None

        # We set public attributes
        self.n_trials = n_trials
        self.path = path if path is not None else join(Paths.TUNING_RECORDS, f"{strftime('%Y%m%d-%H%M%S')}")
        self.save_hps_importance = save_hps_importance
        self.save_parallel_coordinates = save_parallel_coordinates
        self.save_optimization_history = save_optimization_history

        # We make sure that the path given exists
        makedirs(self.path, exist_ok=True)

    def _new_study(self, study_name: str) -> Study:
        """
        Creates a new optuna study

        Args:
            study_name: name of the optuna study

        Returns: study object
        """
        # The metric can be None if the objective is associated to a PetaleEncoder
        if self._objective.metric is None:
            direction = Direction.MINIMIZE
        else:
            direction = self._objective.metric.direction

        return create_study(direction=direction,
                            study_name=study_name,
                            sampler=TPESampler(n_startup_trials=20,
                                               n_ei_candidates=20,
                                               multivariate=True,
                                               constant_liar=True),
                            pruner=NopPruner())

    def _plot_hps_importance_graph(self) -> None:
        """
        Plots the hyperparameters importance graph and save it in an html file

        Returns: None
        """
        # We generate the hyperparameters importance graph with optuna
        fig = plot_param_importances(self._study, evaluator=FanovaImportanceEvaluator(seed=Tuner.HP_IMPORTANCE_SEED))

        # We save the graph
        fig.write_image(join(self.path, Tuner.HPS_IMPORTANCE_FIG))

    def _plot_parallel_coordinates_graph(self) -> None:
        """
        Plots the parallel coordinates graph and save it in an html file
        """

        # We generate the parallel coordinate graph with optuna
        fig = plot_parallel_coordinate(self._study)

        # We save the graph
        fig.write_image(join(self.path, Tuner.PARALLEL_COORD_FIG))

    def _plot_optimization_history_graph(self) -> None:
        """
        Plots the optimization history graph and save it in a html file
        """

        # We generate the optimization history graph with optuna
        fig = plot_optimization_history(self._study)

        # We save the graph
        fig.write_image(join(self.path, Tuner.OPTIMIZATION_HIST_FIG))

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

        Returns: best hyperparameters and hyperparameters' importance

        """
        if self._study is None or self._objective is None:
            raise Exception("study and objective must be defined")

        # We check ray status
        ray_already_init = self._check_ray_status()

        # We perform the optimization
        set_verbosity(FATAL)  # We remove verbosity from loading bar
        self._study.optimize(self._objective, self.n_trials, show_progress_bar=verbose)

        # We save the plots if it is required
        if self.save_hps_importance:
            self._plot_hps_importance_graph()

        if self.save_parallel_coordinates:
            self._plot_parallel_coordinates_graph()

        if self.save_optimization_history:
            self._plot_optimization_history_graph()

        # We extract the best hyperparameters and their importance
        best_hps = self.get_best_hps()
        hps_importance = get_param_importances(self._study,
                                               evaluator=FanovaImportanceEvaluator(seed=Tuner.HP_IMPORTANCE_SEED))

        # We shutdown ray if it has been initialized in this function
        if not ray_already_init:
            ray.shutdown()

        return best_hps, hps_importance

    def update_tuner(self,
                     study_name: str,
                     objective: Objective,
                     saving_path: Optional[str] = None) -> None:
        """
        Sets study and objective protected attributes

        Args:
            study_name: name of the optuna study
            objective: objective function to optimize
            saving_path: path where the tuning details will be stored

        Returns: None
        """
        self._objective = objective
        self._study = self._new_study(study_name)
        self.path = saving_path if saving_path is not None else self.path

    @staticmethod
    def _check_ray_status() -> bool:
        """
        Checks if ray was already initialized and initialize it if it's not

        Returns: true if it was already initialized
        """
        # We initialize ray if it is not initialized yet
        ray_was_init = True
        if not ray.is_initialized():
            ray_was_init = False
            ray.init()

        return ray_was_init


