"""
Authors : Mehdi Mitiche
          Nicolas Raymond

File that contains the class related to the evaluation of the models

"""
import ray

from abc import ABC, abstractmethod
from copy import deepcopy
from numpy.random import seed as np_seed
from os import path, makedirs
from settings.paths import Paths
from sklearn.ensemble import RandomForestClassifier
from src.data.processing.datasets import PetaleNNDataset, PetaleRFDataset, PetaleLinearModelDataset, CustomDataset
from src.data.processing.feature_selection import FeatureSelector
from src.models.models_generation import NNModelGenerator, build_elasticnet
from src.models.nn_models import NNClassifier, NNRegression
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
                 feature_selector: Optional[FeatureSelector] = None,
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
            feature_selector: feature selector object used to proceed to feature selection during nested cross valid
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
        self._feature_selector = feature_selector
        self._feature_selection_count = {feature: 0 for feature in self._dataset.original_data.columns}
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

    def extract_subset(self, records_path: str) -> CustomDataset:
        """
        Executes the feature selection process, save the subset in the protected attributes
        and save a record of the procedure in at the "record_path".

        Args:
            records_path: directory where the feature selection record will be save
        Returns:
        """
        # Creation of subset using feature selection
        if self._feature_selector is not None:
            cont_cols, cat_cols = self._feature_selector(self._dataset, records_path)
            subset = self._dataset.create_subset(cont_cols=cont_cols, cat_cols=cat_cols)
        else:
            subset = deepcopy(self._dataset)

        # Update of feature appearances count
        for feature in subset.original_data.columns:
            self._feature_selection_count[feature] += 1

        return subset

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

            # We update the original datasets masks
            self._dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

            # We create the Recorder object to save the result of this experience
            recorder = Recorder(evaluation_name=self.evaluation_name, index=k,
                                recordings_path=Paths.EXPERIMENTS_RECORDS)

            # We save the saving path
            saving_path = path.join(Paths.EXPERIMENTS_RECORDS, self.evaluation_name, f"Split_{k}")

            # We proceed to feature selection
            subset = self.extract_subset(records_path=saving_path)

            # We record the data count
            for name, mask in [("train_set", train_mask), ("valid_set", valid_mask), ("test_set", test_mask)]:
                mask_length = len(mask) if mask is not None else 0
                recorder.record_data_info(name, mask_length)

            # We update the tuner to perform the hyperparameters optimization
            print(f"\nHyperparameter tuning started - K = {k}\n")
            self._tuner.update_tuner(study_name=f"{self.evaluation_name}_{k}",
                                     objective=self._create_objective(masks=in_masks, subset=subset),
                                     saving_path=saving_path)

            # We perform the hps tuning to get the best hps
            best_hps, hps_importance = self._tuner.tune()

            # We save the hyperparameters
            print(f"\nHyperparameter tuning done - K = {k}\n")
            recorder.record_hyperparameters(best_hps)

            # We save the hyperparameters importance
            recorder.record_hyperparameters_importance(hps_importance)

            # We create a model and a trainer with the best hps
            model, trainer = self._create_model_and_trainer(best_hps)

            # We train our model with the best hps
            print(f"\nFinal model training - K = {k}\n")
            subset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
            trainer.fit(dataset=subset, visualization=True, path=saving_path)

            # We save the trained model
            recorder.record_model(model=trainer.model)

            # We extract x_cont, x_cat and target from the test set
            inputs, targets = trainer.extract_data(subset[test_mask])
            ids = [subset.ids[i] for i in test_mask]

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
    def _create_objective(self, masks: Dict[int, Dict[str, List[int]]], subset: CustomDataset) -> Objective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning
            subset: subset of the original dataset after feature selection

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
                 feature_selector: Optional[FeatureSelector] = None,
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
            feature_selector: feature selector object used to proceed to feature selection during nested cross valid
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """

        # We call parent's constructor
        super().__init__(model_generator=build_elasticnet, dataset=dataset, masks=masks, hps=hps,
                         n_trials=n_trials, optimization_metric=optimization_metric, feature_selector=feature_selector,
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

    def _create_objective(self, masks: Dict[int, Dict[str, List[int]]], subset: PetaleLinearModelDataset
                          ) -> ElasticNetObjective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning
            subset: subset of the original dataset after feature selection

        Returns: objective function
        """
        return ElasticNetObjective(dataset=subset, masks=masks, hps=self._hps,
                                   metric=self.optimization_metric)


class NNEvaluator(Evaluator):
    """
    Object charged to evaluate performances of Neural Network model over multiple splits
    """
    def __init__(self, dataset: PetaleNNDataset, masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], n_trials: int,
                 optimization_metric: Metric, evaluation_metrics: Dict[str, Metric], max_epochs: int,
                 early_stopping: bool, seed: Optional[int] = None,
                 feature_selector: Optional[FeatureSelector] = None,
                 device: Optional[str] = "cpu", evaluation_name: Optional[str] = None,
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
            max_epochs: maximal number of epochs that a trainer can execute with or without early stopping
            early_stopping: True if we want to use early stopping
            seed: random state used for reproducibility
            feature_selector: feature selector object used to proceed to feature selection during nested cross valid
            device: "cpu" or "gpu"
            evaluation_name: name of the results file saved at the recordings_path
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """
        # We call parent's constructor
        super().__init__(model_generator=self._create_model_generator(dataset), dataset=dataset,
                         masks=masks, hps=hps, n_trials=n_trials, optimization_metric=optimization_metric,
                         evaluation_metrics=evaluation_metrics, seed=seed, device=device,
                         feature_selector=feature_selector, evaluation_name=evaluation_name,
                         save_hps_importance=save_hps_importance,
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

    def _create_objective(self, masks: Dict[int, Dict[str, List[int]]], subset: PetaleNNDataset) -> NNObjective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning
            subset: subset of the original dataset after feature selection

        Returns: objective function
        """
        # We update the model generator since some categorical features might
        # have been removed
        self.model_generator.update_cat_sizes(subset.cat_sizes)
        return NNObjective(model_generator=self.model_generator, dataset=subset, masks=masks,
                           hps=self._hps, device=self._device, metric=self.optimization_metric,
                           n_epochs=self._max_epochs, early_stopping=self._early_stopping)

    @staticmethod
    def _create_model_generator(dataset: PetaleNNDataset) -> Callable:
        """
        Creates the object used to generates nn architectures
        Args:
            dataset: PetaleNNDataset

        Returns: Model generator function
        """
        if dataset.classification:
            constructor = NNClassifier
            output_size = len(dataset.original_data[dataset.target].unique())
        else:
            constructor = NNRegression
            output_size = None

        return NNModelGenerator(model_class=constructor,
                                num_cont_col=len(dataset.cont_cols),
                                cat_sizes=dataset.cat_sizes,
                                output_size=output_size)


class RFEvaluator(Evaluator):
    """
    Object charged to evaluate performances of Random Forest classifier model over multiple splits
    """
    def __init__(self, dataset: PetaleRFDataset, masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]], n_trials: int, optimization_metric: Metric,
                 evaluation_metrics: Dict[str, Metric],
                 seed: Optional[int] = None, evaluation_name: Optional[str] = None,
                 feature_selector: Optional[FeatureSelector] = None,
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
            feature_selector: feature selector object used to proceed to feature selection during nested cross valid
            save_hps_importance: True if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: True if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: True if we want to plot the optimization history graph after tuning
        """
        # We call parent's constructor
        super().__init__(model_generator=RandomForestClassifier, dataset=dataset, masks=masks,
                         hps=hps, n_trials=n_trials, optimization_metric=optimization_metric,
                         evaluation_metrics=evaluation_metrics, seed=seed, evaluation_name=evaluation_name,
                         feature_selector=feature_selector, save_hps_importance=save_hps_importance,
                         save_parallel_coordinates=save_parallel_coordinates,
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

    def _create_objective(self, masks: Dict[int, Dict[str, List[int]]], subset: PetaleRFDataset) -> RFObjective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning
            subset: subset of the original dataset after feature selection

        Returns: objective function
        """
        return RFObjective(dataset=subset, masks=masks, hps=self._hps, metric=self.optimization_metric)
