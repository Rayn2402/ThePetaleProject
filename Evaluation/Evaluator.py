"""
Authors : Mehdi Mitiche
          Nicolas Raymond

File that contains the class related to the evaluation of the models

"""
import ray

from abc import ABC, abstractmethod
from Data.Datasets import PetaleNNDataset, PetaleRFDataset
from Hyperparameters.constants import *
from Models.ModelGenerator import NNModelGenerator, RFCModelGenerator
from numpy.random import seed as np_seed
from os import path, makedirs
from Recording.Recorder import NNRecorder, RFRecorder,\
    compare_prediction_recordings, get_evaluation_recap, plot_hyperparameter_importance_chart
from sklearn.ensemble import RandomForestClassifier
from time import strftime
from torch import manual_seed
from torch.nn import Module
from Training.Trainer import NNTrainer, RFTrainer
from Tuning.Tuner import Tuner, NNObjective, RFObjective
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from Utils.score_metrics import Metric


class Evaluator(ABC):
    """
    Abstract class representing the skeleton of the objects used for model evaluation
    """

    def __init__(self, model_generator: Union[NNModelGenerator, RFCModelGenerator],
                 dataset: Union[PetaleNNDataset, PetaleRFDataset], masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], n_trials: int, optimization_metric: Metric,
                 evaluation_metrics: Dict[str, Metric], seed: Optional[int] = None, device: Optional[str] = "cpu",
                 recordings_path: Optional[str] = "Recordings", evaluation_name: Optional[str] = None,
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
            recordings_path: path where recording files will be saved
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """

        # We look if a file with the same evaluation name exists
        if evaluation_name is not None:
            assert not path.exists(path.join(recordings_path, evaluation_name)), \
                "Evaluation with this name already exists"
        else:
            makedirs(recordings_path, exist_ok=True)
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
                            path=recordings_path)

        # We set the public attributes
        self.evaluation_name = evaluation_name
        self.model_generator = model_generator
        self.optimization_metric = optimization_metric
        self.evaluation_metrics = evaluation_metrics
        self.seed = seed
        self.recordings_path = recordings_path

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

        # We initialize ray if it is not initialized yet
        if not ray.is_initialized():
            ray.init()

        # We execute the outer loop
        for k, v in self._masks.items():

            # We extract the masks
            train_mask, valid_mask, test_mask, in_masks = v["train"], v["valid"], v["test"], v["inner"]

            # We create the Recorder object to save the result of this experience
            recorder = self._create_recorder(idx=k)

            # We save the saving path
            saving_path = path.join(self.recordings_path, self.evaluation_name, f"Split_{k}")

            # We record the data count
            for name, mask in [("train_set", train_mask), ("valid_set", valid_mask), ("test_set", test_mask)]:
                recorder.record_data_info(name, len(mask))

            # We update the tuner to perform the hyperparameters optimization
            print(f"\nHyperparameter tuning started - K = {k}\n")
            self._tuner.update_tuner(study_name=f"{self.evaluation_name}_{k}",
                                     objective=self._create_objective(),
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
            recorder.record_model(model=model)

            # We extract x_cont, x_cat and target from the test set
            inputs, targets = trainer.extract_data(self._dataset[test_mask])
            ids = [self._dataset.ids[i] for i in test_mask]

            # We get the predictions
            predictions = trainer.predict(**inputs, log_prob=True)

            if predictions.shape[1] == 1:
                predictions = predictions.flatten()

            # We save the predictions
            recorder.record_predictions(predictions=predictions, ids=ids, target=targets)

            # We save the scores associated to the different evaluation metric
            for metric_name, f in self.evaluation_metrics.items():
                recorder.record_scores(score=f(predictions, targets), metric=metric_name)

            # We save all the data collected in a file
            recorder.generate_file()

            compare_prediction_recordings(evaluations=[self.evaluation_name],
                                          split_index=k, recording_path=self.recordings_path)

        # We save the evaluation recap
        get_evaluation_recap(evaluation_name=self.evaluation_name, recordings_path=self.recordings_path)

        # We save the hyperparameters plot
        plot_hyperparameter_importance_chart(evaluation_name=self.evaluation_name, recordings_path=self.recordings_path)

    @abstractmethod
    def _create_model_and_trainer(self, best_hps: Dict[str, Any]
                                  ) -> Tuple[Union[RandomForestClassifier, Module], Union[NNTrainer, RFTrainer]]:
        """
        Returns a model built according to the best hyperparameters given and a trainer

        Args:
            best_hps: hyperparameters to use in order to build the model

        Returns: model and trainer
        """
        raise NotImplementedError

    @abstractmethod
    def _create_objective(self) -> Union[NNObjective, RFObjective]:
        """
        Creates an adapted objective function to pass to our tuner

        Returns: objective function
        """
        raise NotImplementedError

    @abstractmethod
    def _create_recorder(self, idx: int) -> Union[NNRecorder, RFRecorder]:
        """
        Creates a recorder object adapted to our model

        Args:
            idx: index of outer loop

        Returns: recorder object

        """
        raise NotImplementedError


class NNEvaluator(Evaluator):

    def __init__(self, model_generator: NNModelGenerator, dataset: PetaleNNDataset,
                 masks: Dict[int, Dict[str, List[int]]], hps: Dict[str, Dict[str, Any]], n_trials: int,
                 optimization_metric: Metric, evaluation_metrics: Dict[str, Metric], max_epochs: int,
                 early_stopping: bool, seed: Optional[int] = None, device: Optional[str] = "cpu",
                 recordings_path: Optional[str] = "Recordings", evaluation_name: Optional[str] = None,
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
            max_epochs: maximal number of epochs that a trainer can execute with or without early stopping
            early_stopping: True if we want to use early stopping
            seed: random state used for reproducibility
            device: "cpu" or "gpu"
            recordings_path: path where recording files will be saved
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """

        # We call parent's constructor
        super().__init__(model_generator, dataset, masks, hps, n_trials, optimization_metric,
                         evaluation_metrics, seed, device, recordings_path, evaluation_name,
                         save_hps_importance, save_parallel_coordinates, save_optimization_history)

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
        model = self.model_generator(layers=best_hps[LAYERS], dropout=best_hps[DROPOUT],
                                     activation=best_hps[ACTIVATION])
        trainer = NNTrainer(model=model, metric=self.optimization_metric, lr=best_hps[LR],
                            batch_size=best_hps[BATCH_SIZE], weight_decay=best_hps[WEIGHT_DECAY],
                            epochs=self._max_epochs, early_stopping=self._early_stopping,
                            device=self._device, in_trial=False)

        return model, trainer

    def _create_objective(self) -> NNObjective:
        """
        Creates an objective function adapted to neural networks

        Returns: objective function
        """
        return NNObjective(model_generator=self.model_generator, dataset=self._dataset, masks=self._masks,
                           hps=self._hps, device=self._device, metric=self.optimization_metric,
                           n_epochs=self._max_epochs, early_stopping=self._early_stopping)

    def _create_recorder(self, idx: int) -> NNRecorder:
        """
        Creates a recorder adapted to neural networks

        Args:
            idx: index of outer loop

        Returns: recorder
        """
        return NNRecorder(evaluation_name=self.evaluation_name, index=idx, recordings_path=self.recordings_path)


class RFEvaluator(Evaluator):

    def __init__(self, model_generator: RFCModelGenerator, dataset: PetaleRFDataset,
                 masks: Dict[int, Dict[str, List[int]]], hps: Dict[str, Dict[str, Any]],
                 n_trials: int, optimization_metric: Metric, evaluation_metrics: Dict[str, Metric],
                 seed: Optional[int] = None, device: Optional[str] = "cpu",
                 recordings_path: Optional[str] = "Recordings", evaluation_name: Optional[str] = None,
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
            recordings_path: path where recording files will be saved
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """
        # We call parent's constructor
        super().__init__(model_generator, dataset, masks, hps, n_trials, optimization_metric,
                         evaluation_metrics, seed, device, recordings_path, evaluation_name,
                         save_hps_importance, save_parallel_coordinates, save_optimization_history)

    def _create_model_and_trainer(self, best_hps: Dict[str, Any]) -> Tuple[RandomForestClassifier, RFTrainer]:
        """
        Creates a random forest classifier and its trainer according to the best hyperparameters

        Args:
            best_hps: dictionary of hyperparameters

        Returns: model and trainer
        """
        model = self.model_generator(n_estimators=best_hps[N_ESTIMATORS], max_features=best_hps[MAX_FEATURES],
                                     max_depth=best_hps[MAX_DEPTH], max_samples=best_hps[MAX_SAMPLES])

        trainer = RFTrainer(model=model, metric=self.optimization_metric)

        return model, trainer

    def _create_objective(self) -> RFObjective:
        """
        Creates an objective function adapted to random forest classifier

        Returns: objective function
        """
        return RFObjective(model_generator=self.model_generator, dataset=self._dataset,
                           masks=self._masks, hps=self._hps, device=self._device, metric=self.optimization_metric)

    def _create_recorder(self, idx: int) -> RFRecorder:
        """
        Creates a recorder adapted to random forest classifier

        Args:
            idx: index of outer loop

        Returns: recorder
        """
        return RFRecorder(evaluation_name=self.evaluation_name, index=idx, recordings_path=self.recordings_path)
