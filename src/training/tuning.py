"""
Authors : Mehdi Mitiche
          Nicolas Raymond

Files that contains the logic related to hyper parameters tuning

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
from optuna.visualization import plot_param_importances, plot_parallel_coordinate, plot_optimization_history
from os import makedirs
from os.path import join
from settings.paths import Paths
from src.data.processing.datasets import PetaleDataset
from src.models.abstract_models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.training.enums import *
from src.utils.score_metrics import Metric
from src.utils.hyperparameters import Distribution, CategoricalHP, NumericalContinuousHP, NumericalIntHP, HP
from time import strftime
from torch import mean, tensor
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


class Objective:
    """
    Objective function to use with the tuner
    """
    def __init__(self, dataset: PetaleDataset, masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], fixed_params: Dict[str, Any], metric: Metric,
                 model_constructor: Callable, gpu_device: bool = False):
        """
        Sets protected and public attributes

        Args:
            dataset: custom dataset containing all the data needed for our evaluations
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            metric: callable metric we want to optimize with hyperparameters (not used for backpropagation)
        """
        # We validate the given hyperparameters
        for hp in model_constructor.get_hps():
            assert hp.name in list(hps.keys()), f"'{hp}' is missing from hps dictionary"

        # We set protected attributes
        self._dataset = dataset
        self._fixed_params = fixed_params
        self._hps = hps
        self._masks = masks
        self._metric = metric
        self._model_constructor = model_constructor
        self._getters = {}
        self._define_getters()

        # We set protected parallel method
        self._run_parallel_evaluations = self._build_parallel_process(gpu_device)

    @property
    def metric(self):
        return self._metric

    def __call__(self, trial: Trial) -> float:
        """
        Extracts hyperparameters suggested by optuna and executes parallel evaluation
        of the hyperparameters set

        Args:
            trial: optuna trial

        Returns: score associated to the set of hyperparameters
        """
        # We extract hyperparameters suggestions
        suggested_hps = {k: f(trial) for k, f in self._getters.items()}

        # We execute parallel evaluations
        futures = [self._run_parallel_evaluations.remote(masks=m, hps=suggested_hps) for k, m in self._masks.items()]
        scores = ray.get(futures)

        # We take the mean of the scores
        return mean(tensor(scores)).item()

    def _define_getters(self) -> None:
        """
        Defines the different optuna sampling function for each hyperparameter
        """
        for hp in self._model_constructor.get_hps():

            # We check if a value was predefined for this hyperparameter
            if Range.VALUE in self._hps[hp.name].keys():
                self._getters[hp.name] = self._build_constant_getter(hp)

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

    def _build_numerical_cont_getter(self, hp):
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
        Builds the function that run parallel evaluations for each set of hyperparameters
        and return the score

        Args:
            gpu_device: True indicates that we want to use a gpu

        Returns: function
        """
        @ray.remote(num_gpus=int(gpu_device))
        def run_parallel_evaluations(masks: Dict[str, List[int]], hps: Dict[str, Any]) -> float:
            """
            Train a single model using given masks and given hyperparameters
            Args:
                masks: dictionary with list of integers for train, valid and test mask
                hps: dictionary with hyperparameters to give to the model constructor

            Returns: metric score
            """
            # We extract masks
            train_idx, valid_idx, test_idx = masks['train'], masks['valid'], masks['test']

            # We create a copy of the current dataset and update its masks
            dts = deepcopy(self._dataset)
            dts.update_masks(train_mask=train_idx, valid_mask=valid_idx, test_mask=test_idx)

            # We build a model using hps and fixed params (PetaleRegressor or PetaleClassifier)
            model = self._model_constructor(**hps, **self._fixed_params)

            # We train the model
            model.fit(dts)

            # If the model is a classifier we find its optimal threshold
            if isinstance(model, PetaleBinaryClassifier):
                model.find_optimal_threshold(dataset=dts, metric=self._metric)
                pred = model.predict_proba(dataset=dts)
                _, y, _ = dts[dts.test_mask]
                score = self._metric(pred, y, thresh=model.thresh)
            else:
                pred = model.predict(dataset=dts)
                _, y, _ = dts[dts.test_mask]
                score = self._metric(pred, y)

            return score

        return run_parallel_evaluations

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
    Base of all objects used for hyperparameter tuning
    """
    def __init__(self, n_trials: int, study_name: Optional[str] = None,
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
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
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
        return create_study(direction=self._objective.metric.direction,
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

        Returns: best hyperparameters and hyperparameters' importance

        """
        assert (self._study is not None and self._objective is not None), "study and objective must be defined"

        # We check ray status
        ray_already_init = self._check_ray_status()

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

        # We shutdown ray it has been initialized in this function
        if not ray_already_init:
            ray.shutdown()

        return best_hps, hps_importance

    def update_tuner(self, study_name: str, objective: Objective,
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

        Returns: True if yes
        """
        # We initialize ray if it is not initialized yet
        ray_was_init = True
        if not ray.is_initialized():
            ray_was_init = False
            ray.init()

        return ray_was_init


