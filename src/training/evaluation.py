"""
Filename: evaluation.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to define the Evaluator class in charge
             of comparing models against each other

Date of last modification : 2022/03/30
"""
import ray

from copy import deepcopy
from json import load
from numpy.random import seed as np_seed
from os import makedirs, path
from pandas import DataFrame
from settings.paths import Paths
from src.data.extraction.constants import PARTICIPANT
from src.data.processing.datasets import MaskType, PetaleDataset
from src.data.processing.feature_selection import FeatureSelector
from src.models.abstract_models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.recording.constants import PREDICTION, RECORDS_FILE, TEST_RESULTS, TRAIN_RESULTS, VALID_RESULTS
from src.recording.recording import Recorder, compare_prediction_recordings, \
    get_evaluation_recap, plot_feature_importance_charts, plot_hps_importance_chart
from src.training.tuning import Objective, Tuner
from src.utils.score_metrics import Metric
from time import strftime
from torch import is_tensor, from_numpy, manual_seed
from typing import Any, Callable, Dict, List, Optional, Union


class Evaluator:
    """
    Object in charge of evaluating a model over multiple different data splits.
    """
    def __init__(self,
                 model_constructor: Callable,
                 dataset: PetaleDataset,
                 masks: Dict[int, Dict[str, List[int]]],
                 hps: Dict[str, Dict[str, Any]],
                 n_trials: int,
                 evaluation_metrics: List[Metric],
                 fixed_params: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None,
                 gpu_device: bool = False,
                 evaluation_name: Optional[str] = None,
                 feature_selector: Optional[FeatureSelector] = None,
                 fixed_params_update_function: Optional[Callable] = None,
                 save_hps_importance: Optional[bool] = False,
                 save_parallel_coordinates: Optional[bool] = False,
                 save_optimization_history: Optional[bool] = False,
                 pred_path: Optional[str] = None):
        """
        Set protected and public attributes

        Args:
            model_constructor: callable object used to generate a model according to a set of hyperparameters
            dataset: custom dataset containing the whole learning dataset needed for our evaluations
            masks: dict with list of idx to use as train, valid and test masks
            hps: dictionary with information on the hyperparameters we want to tune
            fixed_params: dictionary with parameters used by the model constructor for building model
            n_trials: number of hyperparameters sets sampled within each inner validation loop
            evaluation_metrics: list of metrics to evaluate on models built for each outer split.
                                The last one is used for hyperparameter optimization
            seed: random state used for reproducibility
            gpu_device: True if we want to use the gpu
            evaluation_name: name of the results file saved at the recordings_path
            feature_selector: object used to proceed to feature selection during nested cross valid
            fixed_params_update_function: function that updates fixed params dictionary from PetaleSubset after
                                          feature selection. Might be necessary for model with entity embedding.
            save_hps_importance: true if we want to plot the hyperparameters importance graph after tuning
            save_parallel_coordinates: true if we want to plot the parallel coordinates graph after tuning
            save_optimization_history: true if we want to plot the optimization history graph after tuning
            pred_path: if given, the path will be used to load predictions from another experiment and
                       include them within actual features.
        """

        # We look if a file with the same evaluation name exists
        if evaluation_name is not None:
            if path.exists(path.join(Paths.EXPERIMENTS_RECORDS, evaluation_name)):
                raise ValueError("evaluation with this name already exists")
        else:
            makedirs(Paths.EXPERIMENTS_RECORDS, exist_ok=True)
            evaluation_name = f"{strftime('%Y%m%d-%H%M%S')}"

        # We set protected attributes
        self._dataset = dataset
        self._gpu_device = gpu_device
        self._feature_selector = feature_selector
        self._feature_selection_count = {feature: 0 for feature in self._dataset.original_data.columns}
        self._fixed_params = fixed_params if fixed_params is not None else {}
        self._hps = hps
        self._masks = masks
        self._pred_path = pred_path
        self._hp_tuning = (n_trials > 0)
        self._tuner = Tuner(n_trials=n_trials,
                            save_hps_importance=save_hps_importance,
                            save_parallel_coordinates=save_parallel_coordinates,
                            save_optimization_history=save_optimization_history,
                            path=Paths.EXPERIMENTS_RECORDS)

        # We set the public attributes
        self.evaluation_name = evaluation_name
        self.model_constructor = model_constructor
        self.evaluation_metrics = evaluation_metrics
        self.seed = seed

        # We set the fixed params update method
        if fixed_params_update_function is not None:
            self._update_fixed_params = fixed_params_update_function
        else:
            self._update_fixed_params = lambda _: self._fixed_params

    def _extract_subset(self,
                        records_path: str,
                        recorder: Recorder) -> PetaleDataset:
        """
        Executes the feature selection process and save a record of
        the procedure at the "records_path".

        Args:
            records_path: directory where the feature selection record will be save

        Returns: Columns subset of the current dataset (PetaleDataset)
        """
        # Creation of subset using feature selection
        if self._feature_selector is not None:
            cont_cols, cat_cols, fi_dict = self._feature_selector(self._dataset, records_path, return_imp=True)
            recorder.record_features_importance(fi_dict)
            subset = self._dataset.create_subset(cont_cols=cont_cols, cat_cols=cat_cols)
        else:
            subset = deepcopy(self._dataset)

        # Update of feature appearances count
        for feature in subset.original_data.columns:
            self._feature_selection_count[feature] += 1

        return subset

    def evaluate(self) -> None:
        """
        Performs nested subsampling validations to evaluate a model and saves results
        in specific files using a recorder

        Returns: None

        """
        # We set the seed for the nested subsampling valid procedure
        if self.seed is not None:
            np_seed(self.seed)
            manual_seed(self.seed)

        # We initialize ray
        ray.init()

        # We execute the outer loop
        for k, v in self._masks.items():

            # We extract the masks
            train_mask, valid_mask = v[MaskType.TRAIN], v[MaskType.VALID]
            test_mask, in_masks = v[MaskType.TEST], v[MaskType.INNER]

            # We update the dataset's masks
            self._dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

            # We create the Recorder object to save the result of this experience
            recorder = Recorder(evaluation_name=self.evaluation_name,
                                index=k,
                                recordings_path=Paths.EXPERIMENTS_RECORDS)

            # We save the saving path
            saving_path = path.join(Paths.EXPERIMENTS_RECORDS, self.evaluation_name, f"Split_{k}")

            # We proceed to feature selection
            subset = self._extract_subset(records_path=saving_path, recorder=recorder)

            # We include predictions from another experiment if needed
            if self._pred_path is not None:
                subset = self._load_predictions(split_number=k, subset=subset)

            # We update the fixed parameters according to the subset
            self._fixed_params = self._update_fixed_params(subset)

            # We record the data count
            for name, mask in [("train_set", train_mask), ("valid_set", valid_mask), ("test_set", test_mask)]:
                mask_length = len(mask) if mask is not None else 0
                recorder.record_data_info(name, mask_length)

            # We update the tuner to perform the hyperparameters optimization
            if self._hp_tuning:

                print(f"\nHyperparameter tuning started - K = {k}\n")

                # We update the tuner
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
            else:
                best_hps = {}

            # We create a model with the best hps
            model = self.model_constructor(**best_hps, **self._fixed_params)

            # We train our model with the best hps
            print(f"\nFinal model training - K = {k}\n")
            subset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
            model.fit(dataset=subset)

            # We save plots associated to training
            if hasattr(model, 'plot_evaluations'):
                model.plot_evaluations(save_path=saving_path)

            # We save the trained model
            model.save_model(path=saving_path)

            # We get the predictions and save the evaluation metric scores
            self._record_scores_and_pred(model, recorder, subset)

            # We save all the data collected in one file
            recorder.generate_file()

            # We generate a plot that compares predictions to ground_truth
            if not model.is_encoder():
                compare_prediction_recordings(evaluations=[self.evaluation_name],
                                              split_index=k,
                                              recording_path=Paths.EXPERIMENTS_RECORDS)

        # We save the evaluation recap
        get_evaluation_recap(evaluation_name=self.evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)

        # We save the features importance charts
        if self._feature_selector is not None:
            plot_feature_importance_charts(evaluation_name=self.evaluation_name,
                                           recordings_path=Paths.EXPERIMENTS_RECORDS)

        # We save the hyperparameters importance chart
        if self._hp_tuning:
            plot_hps_importance_chart(evaluation_name=self.evaluation_name,
                                      recordings_path=Paths.EXPERIMENTS_RECORDS)
        # We shutdown ray
        ray.shutdown()

    def _create_objective(self,
                          masks: Dict[int, Dict[str, List[int]]],
                          subset: PetaleDataset) -> Objective:
        """
        Creates an adapted objective function to pass to our tuner

        Args:
            masks: inner masks for hyperparameters tuning
            subset: subset of the original dataset after feature selection

        Returns: objective function
        """
        metric = self.evaluation_metrics[-1] if len(self.evaluation_metrics) > 0 else None
        return Objective(dataset=subset,
                         masks=masks,
                         hps=self._hps,
                         fixed_params=self._fixed_params,
                         metric=metric,
                         model_constructor=self.model_constructor,
                         gpu_device=self._gpu_device)

    def _load_predictions(self,
                          split_number: int,
                          subset: PetaleDataset) -> PetaleDataset:
        """
        Loads prediction in a given path and includes them as a feature within the dataset

        Args:
            split_number: split for which we got to load predictions
            subset: actual dataset

        Returns: updated dataset
        """

        # Loading of records
        with open(path.join(self._pred_path, f"Split_{split_number}", RECORDS_FILE), "r") as read_file:
            data = load(read_file)

        # We check the format of predictions
        random_pred = list(data[TRAIN_RESULTS].values())[0][PREDICTION]
        if "[" not in random_pred:

            # Saving of the number of predictions columns
            nb_pred_col = 1

            # Creation of the conversion function to extract predictions from strings
            def convert(x: str) -> List[float]:
                return [float(x)]
        else:

            # Saving of the number of predictions columns
            nb_pred_col = len(random_pred[1:-1].split(','))

            # Creation of the conversion function to extract predictions from strings
            def convert(x: str) -> List[float]:
                return [float(a) for a in x[1:-1].split(',')]

        # Extraction of predictions
        pred = {}
        for section in TRAIN_RESULTS, TEST_RESULTS, VALID_RESULTS:
            pred = {**pred, **{p_id: [p_id, *convert(v[PREDICTION])] for p_id, v in data[section].items()}}

        # Creation of pandas dataframe
        pred_col_names = [f'pred{i}' for i in range(nb_pred_col)]
        df = DataFrame.from_dict(pred, orient='index', columns=[PARTICIPANT, *pred_col_names])

        # Creation of new augmented dataset
        return subset.create_superset(data=df, categorical=False)

    def _record_scores_and_pred(self,
                                model: Union[PetaleBinaryClassifier, PetaleRegressor],
                                recorder: Recorder,
                                subset: PetaleDataset) -> None:
        """
        Records the scores associated to train and test set
        and also saves the prediction linked to each individual

        Args:
            model: model trained with best found hyperparameters
            recorder: object recording information about splits evaluations
            subset: dataset with remaining features from feature selection

        Returns: None
        """
        # If the model is a classifier
        if subset.classification:

            # We find the optimal threshold and save it
            model.find_optimal_threshold(dataset=subset, metric=self.evaluation_metrics[-1])
            recorder.record_data_info('thresh', str(model.thresh))

            for mask, mask_type in [(subset.train_mask, MaskType.TRAIN),
                                    (subset.test_mask, MaskType.TEST),
                                    (subset.valid_mask, MaskType.VALID)]:

                if len(mask) > 0:

                    # We compute prediction
                    pred = model.predict_proba(subset, mask)

                    # We extract ids and targets
                    ids = [subset.ids[i] for i in mask]
                    _, y, _ = subset[mask]

                    # We record all metric scores
                    for metric in self.evaluation_metrics:
                        recorder.record_scores(score=metric(pred, y, thresh=model.thresh),
                                               metric=metric.name, mask_type=mask_type)

                    if not is_tensor(pred):
                        pred = from_numpy(pred)

                    # We get the final predictions from the soft predictions
                    pred = (pred >= model.thresh).long()

                    # We save the predictions
                    recorder.record_predictions(predictions=pred, ids=ids, targets=y, mask_type=mask_type)

        else:   # If instead the model is for regression
            for mask, mask_type in [(subset.train_mask, MaskType.TRAIN),
                                    (subset.test_mask, MaskType.TEST),
                                    (subset.valid_mask, MaskType.VALID)]:

                if len(mask) > 0:

                    # We extract ids and targets
                    ids = [subset.ids[i] for i in mask]
                    _, y, _ = subset[mask]

                    # We get the real-valued predictions
                    pred = model.predict(subset, mask)

                    # We record all metric scores
                    for metric in self.evaluation_metrics:
                        recorder.record_scores(score=metric(pred, y), metric=metric.name, mask_type=mask_type)

                    # We save the predictions
                    recorder.record_predictions(predictions=pred, ids=ids, targets=y, mask_type=mask_type)
