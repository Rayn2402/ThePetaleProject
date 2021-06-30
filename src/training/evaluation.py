"""
Authors : Mehdi Mitiche
          Nicolas Raymond

File that contains the class related to the evaluation of the models

"""
import ray

from abc import ABC, abstractmethod
from numpy.random import seed as np_seed
from os import path, makedirs
from settings.paths import Paths
from sklearn.ensemble import RandomForestClassifier
from src.data.processing.datasets import PetaleNNDataset, PetaleRFDataset, PetaleLinearModelDataset, CustomDataset
from src.models.models_generation import NNModelGenerator, build_elasticnet
from src.recording.recording import Recorder, compare_prediction_recordings, \
    get_evaluation_recap, plot_hyperparameter_importance_chart
from src.training.enums import *
from src.training.training import NNTrainer, RFTrainer, ElasticNetTrainer, Trainer
from src.training.tuning import Tuner, NNObjective, RFObjective, ElasticNetObjective, Objective
from src.utils.score_metrics import Metric
from time import strftime
from torch import manual_seed
from torch.nn import Module
from typing import Any, Callable, Dict, List, Optional, Tuple


class Evaluator(ABC):
    """
    Abstract class representing the skeleton of the objects used for model evaluation
    """

    def __init__(self, model_generator: Callable,
                 dataset: CustomDataset, masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], n_trials: int, optimization_metric: Metric,
                 evaluation_metrics: Dict[str, Metric], seed: Optional[int] = None,
                 device: Optional[str] = "cpu", evaluation_name: Optional[str] = None,
                 save_hps_importance: Optional[bool] = False,
                 save_parallel_coordinates: Optional[bool] = False,
                 save_optimization_history: Optional[bool] = False):
        """
        Set protected and public attributes of the abstract class

        Args:
            model_generator: callable object used to generate a model according to a set of hyperparameters
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            n_trials: number of hyperparameters sets sampled within each inner validation loop
            optimization_metric: function that hyperparameters must optimize
            evaluation_metrics: dict where keys are names of metrics and values are functions
                                that will be used to calculate the score of the associated metric
                                on the test sets of the outer loops
            seed: random state used for reproducibility
            device: "cpu" or "gpu"
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """

        # We look if a file with the same evaluation name exists
        if evaluation_name is not None:
            assert not path.exists(path.join(Paths.EXPERIMENTS_RECORDS, evaluation_name)), \
                "Evaluation with this name already exists"
        else:
            makedirs(Paths.EXPERIMENTS_RECORDS, exist_ok=True)
            evaluation_name = f"{strftime('%Y%m%d-%H%M%S')}"

        # We check the availability of the device choice
        assert device == "cpu" or device == "gpu", "Device must be 'cpu' or 'gpu'"

        # We set protected attributes
        self._dataset = dataset
        self._device = device
        self._hps = hps
        self._masks = masks
        self._tuner = Tuner(n_trials=n_trials,
                            save_hps_importance=save_hps_importance,
                            save_parallel_coordinates=save_parallel_coordinates,
                            save_optimization_history=save_optimization_history,
                            path=Paths.EXPERIMENTS_RECORDS)

        # We set the public attributes
        self.evaluation_name = evaluation_name
        self.model_generator = model_generator
        self.optimization_metric = optimization_metric
        self.evaluation_metrics = evaluation_metrics
        self.seed = seed

    def nested_cross_valid(self) -> None:
        """
        Performs nested subsampling validations to evaluate a model and saves results
        in specific files using a recorder

        Returns: None

        """

        # We set the seed for the nested cross valid procedure
        if self.seed is not None:
            np_seed(self.seed)
            manual_seed(self.seed)

        # We initialize ray
        ray.init()

        # We execute the outer loop
        for k, v in self._masks.items():

            # We extract the masks
            train_mask, valid_mask, test_mask, in_masks = v["train"], v["valid"], v["test"], v["inner"]

            # We create the Recorder object to save the result of this experience
            recorder = Recorder(evaluation_name=self.evaluation_name, index=k,
                                recordings_path=Paths.EXPERIMENTS_RECORDS)

            # We save the saving path
            saving_path = path.join(Paths.EXPERIMENTS_RECORDS, self.evaluation_name, f"Split_{k}")

            # We record the data count
            for name, mask in [("train_set", train_mask), ("valid_set", valid_mask), ("test_set", test_mask)]:
                mask_length = len(mask) if mask is not None else 0
                recorder.record_data_info(name, mask_length)

            # We update the tuner to perform the hyperparameters optimization
            print(f"\nHyperparameter tuning started - K = {k}\n")
            self._tuner.update_tuner(study_name=f"{self.evaluation_name}_{k}",
                                     objective=self._create_objective(masks=in_masks),
                                     saving_path=saving_path)

            # We perform the hyper parameters tuning to get the best hyper parameters
            best_hps, hps_importance = self._tuner.tune()

            # We save the hyperparameters
            print(f"\nHyperparameter tuning done - K = {k}\n")
            recorder.record_hyperparameters(best_hps)

            # We save the hyperparameters importance
            recorder.record_hyperparameters_importance(hps_importance)

            # We create a model and a trainer with the best hyper parameters
            model, trainer = self._create_model_and_trainer(best_hps)

            # We train our model with the best hyper parameters
            print(f"\nFinal model training - K = {k}\n")
            self._dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
            trainer.fit(dataset=self._dataset, visualization=True, path=saving_path)

            # We save the trained model
            recorder.record_model(model=trainer.model)

            # We extract x_cont, x_cat and target from the test set
            inputs, targets = trainer.extract_data(self._dataset[test_mask])
            ids = [self._dataset.ids[i] for i in test_mask]

            # We get the predictions
            predictions = trainer.predict(**inputs, log_prob=True)

            # We save the predictions
            recorder.record_predictions(predictions=predictions, ids=ids, target=targets)

            # We save the scores associated to the different evaluation metric
            for metric_name, f in self.evaluation_metrics.items():
                recorder.record_scores(score=f(predictions, targets), metric=metric_name)

            # We save all the data collected in a file
            recorder.generate_file()

            compare_prediction_recordings(evaluations=[self.evaluation_name],
                                          split_index=k, recording_path=Paths.EXPERIMENTS_RECORDS)

        # We save the evaluation recap
        get_evaluation_recap(evaluation_name=self.evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)

        # We save the hyperparameters plot
        plot_hyperparameter_importance_chart(evaluation_name=self.evaluation_name,
                                             recordings_path=Paths.EXPERIMENTS_RECORDS)

        # We shutdown ray
        ray.shutdown()

    @abstractmethod
    def _create_model_and_trainer(self, best_hps: Dict[str, Any]) -> Tuple[Callable, Trainer]:
        """
        Returns a model built according to the best hyperparameters given and a trainer

        Args:
            best_hps: hyperparameters to use in order to build the model

        Returns: model and trainer
        """
        raise NotImplementedError

    @abstractmethod
    def _create_objective(self, masks: Dict[int, Dict[str, List[int]]]) -> Objective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning

        Returns: objective function
        """
        raise NotImplementedError


class ElasticNetEvaluator(Evaluator):
    """
    Object charged to evaluate performances of ElasticNet model over multiple splits
    """
    def __init__(self, dataset: PetaleLinearModelDataset, masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], n_trials: int, optimization_metric: Metric,
                 evaluation_metrics: Dict[str, Metric], seed: Optional[int] = None,
                 evaluation_name: Optional[str] = None, save_hps_importance: Optional[bool] = False,
                 save_parallel_coordinates: Optional[bool] = False, save_optimization_history: Optional[bool] = False):
        """
        Sets protected and public attributes of the class

        Args:
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            n_trials: number of hyperparameters sets sampled within each inner validation loop
            optimization_metric: function that hyperparameters must optimize
            evaluation_metrics: dict where keys are names of metrics and values are functions
                                that will be used to calculate the score of the associated metric
                                on the test sets of the outer loops
            seed: random state used for reproducibility
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """

        # We call parent's constructor
        super().__init__(model_generator=build_elasticnet, dataset=dataset, masks=masks, hps=hps,
                         n_trials=n_trials, optimization_metric=optimization_metric,
                         evaluation_metrics=evaluation_metrics, seed=seed, evaluation_name=evaluation_name,
                         save_hps_importance=save_hps_importance, save_parallel_coordinates=save_parallel_coordinates,
                         save_optimization_history=save_optimization_history)

    def _create_model_and_trainer(self, best_hps: Dict[str, Any]) -> Tuple[Callable, ElasticNetTrainer]:
        """
        Returns a model built according to the best hyperparameters given and a trainer

        Args:
            best_hps: hyperparameters to use in order to build the model

        Returns: model and trainer
        """
        model = self.model_generator(alpha=best_hps[ElasticNetHP.ALPHA], beta=best_hps[ElasticNetHP.BETA])
        trainer = ElasticNetTrainer(model=model, metric=self.optimization_metric)

        return model, trainer

    def _create_objective(self, masks: Dict[int, Dict[str, List[int]]]) -> ElasticNetObjective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning

        Returns: objective function
        """
        return ElasticNetObjective(dataset=self._dataset, masks=masks, hps=self._hps,
                                   metric=self.optimization_metric)


class NNEvaluator(Evaluator):
    """
    Object charged to evaluate performances of Neural Network model over multiple splits
    """
    def __init__(self, model_generator: NNModelGenerator, dataset: PetaleNNDataset,
                 masks: Dict[int, Dict[str, List[int]]], hps: Dict[str, Dict[str, Any]], n_trials: int,
                 optimization_metric: Metric, evaluation_metrics: Dict[str, Metric], max_epochs: int,
                 early_stopping: bool, seed: Optional[int] = None,
                 device: Optional[str] = "cpu", evaluation_name: Optional[str] = None,
                 save_hps_importance: Optional[bool] = False,
                 save_parallel_coordinates: Optional[bool] = False,
                 save_optimization_history: Optional[bool] = False):

        """
        Sets protected and public attributes of the class

        Args:
            model_generator: callable object used to generate a model according to a set of hyperparameters
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            n_trials: number of hyperparameters sets sampled within each inner validation loop
            optimization_metric: function that hyperparameters must optimize
            evaluation_metrics: dict where keys are names of metrics and values are functions
                                that will be used to calculate the score of the associated metric
                                on the test sets of the outer loops
            max_epochs: maximal number of epochs that a trainer can execute with or without early stopping
            early_stopping: True if we want to use early stopping
            seed: random state used for reproducibility
            device: "cpu" or "gpu"
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """
        # We call parent's constructor
        super().__init__(model_generator=model_generator, dataset=dataset, masks=masks,
                         hps=hps, n_trials=n_trials, optimization_metric=optimization_metric,
                         evaluation_metrics=evaluation_metrics, seed=seed, device=device,
                         evaluation_name=evaluation_name, save_hps_importance=save_hps_importance,
                         save_parallel_coordinates=save_parallel_coordinates,
                         save_optimization_history=save_optimization_history)

        # We set other protected attribute
        self._max_epochs = max_epochs
        self._early_stopping = early_stopping

    def _create_model_and_trainer(self, best_hps: Dict[str, Any]) -> Tuple[Module, NNTrainer]:
        """
        Creates a neural networks and a its trainer using the best hyperparameters

        Args:
            best_hps: dictionary of hyperparameters

        Returns: model and trainer
        """
        model = self.model_generator(layers=best_hps[NeuralNetsHP.LAYERS],
                                     dropout=best_hps[NeuralNetsHP.DROPOUT],
                                     activation=best_hps[NeuralNetsHP.ACTIVATION])

        trainer = NNTrainer(model=model, metric=self.optimization_metric, lr=best_hps[NeuralNetsHP.LR],
                            batch_size=best_hps[NeuralNetsHP.BATCH_SIZE],
                            epochs=self._max_epochs, early_stopping=self._early_stopping,
                            device=self._device, in_trial=False)

        return model, trainer

    def _create_objective(self, masks: Dict[int, Dict[str, List[int]]]) -> NNObjective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning

        Returns: objective function
        """
        return NNObjective(model_generator=self.model_generator, dataset=self._dataset, masks=masks,
                           hps=self._hps, device=self._device, metric=self.optimization_metric,
                           n_epochs=self._max_epochs, early_stopping=self._early_stopping)


class RFEvaluator(Evaluator):
    """
    Object charged to evaluate performances of Random Forest classifier model over multiple splits
    """
    def __init__(self, dataset: PetaleRFDataset, masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], n_trials: int, optimization_metric: Metric,
                 evaluation_metrics: Dict[str, Metric],
                 seed: Optional[int] = None, evaluation_name: Optional[str] = None,
                 save_hps_importance: Optional[bool] = False,
                 save_parallel_coordinates: Optional[bool] = False,
                 save_optimization_history: Optional[bool] = False):
        """
        Sets protected and public attributes of the class

        Args:
            dataset: custom dataset containing the whole learning dataset needed for our evaluation
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            n_trials: number of hyperparameters sets sampled within each inner validation loop
            optimization_metric: function that hyperparameters must optimize
            evaluation_metrics: dict where keys are names of metrics and values are functions
                                that will be used to calculate the score of the associated metric
                                on the test sets of the outer loops
            seed: random state used for reproducibility
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """
        # We call parent's constructor
        super().__init__(model_generator=RandomForestClassifier, dataset=dataset, masks=masks,
                         hps=hps, n_trials=n_trials, optimization_metric=optimization_metric,
                         evaluation_metrics=evaluation_metrics, seed=seed, evaluation_name=evaluation_name,
                         save_hps_importance=save_hps_importance, save_parallel_coordinates=save_parallel_coordinates,
                         save_optimization_history=save_optimization_history)

    def _create_model_and_trainer(self, best_hps: Dict[str, Any]) -> Tuple[RandomForestClassifier, RFTrainer]:
        """
        Creates a random forest classifier and its trainer according to the best hyperparameters

        Args:
            best_hps: dictionary of hyperparameters

        Returns: model and trainer
        """
        model = self.model_generator(n_estimators=best_hps[RandomForestsHP.N_ESTIMATORS],
                                     max_features=best_hps[RandomForestsHP.MAX_FEATURES],
                                     max_depth=best_hps[RandomForestsHP.MAX_DEPTH],
                                     max_samples=best_hps[RandomForestsHP.MAX_SAMPLES])

        trainer = RFTrainer(model=model, metric=self.optimization_metric)

        return model, trainer

    def _create_objective(self, masks: Dict[int, Dict[str, List[int]]]) -> RFObjective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning

        Returns: objective function
        """
        return RFObjective(dataset=self._dataset, masks=masks, hps=self._hps, metric=self.optimization_metric)
