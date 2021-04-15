"""
Authors : Mehdi Mitiche

File that contains the class related to the evaluation of the models

"""
from Trainer.Trainer import NNTrainer, RFTrainer
from Tuner.Tuner import NNTuner, RFTuner
from torch import manual_seed
from numpy.random import seed as np_seed
from Hyperparameters.constants import *
from Recorder.Recorder import NNRecorder, Recorder, get_evaluation_recap, plot_hyperparameter_importance_chart

import ray
import time
from os import path, mkdir, makedirs
from shutil import rmtree


class Evaluator:
    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, optimization_metric,
                 evaluation_metrics, k, l=1, direction="minimize", seed=None, get_hyperparameters_importance=False,
                 get_parallel_coordinate=False, get_optimization_history=False, device="cpu", parallelism=True,
                 recordings_path=""):
        """
        Class that will be responsible of the evaluation of the model

        :param evaluation_name: String that represents the name of the evaluation
        :param model_generator: instance of the ModelGenerator class that will be responsible of generating the model
        :param sampler: A sampler object that will be called to perform the stratified sampling to get all the train
        and test set for both the inner and the outer training
        :param hyper_params: dictionary containing information of the hyper parameter we want to tune
        :param optimization_metric: a function that takes the output of the model and the target and returns  the metric
         we want to optimize
        :param evaluation_metrics:  dictonary where keys represent name of metrics and values represent
                                    the function that will be used to calculate the score of
                                    the associated metric
        :param k: Number of folds in the outer cross validation
        :param l: Number of folds in the inner cross validation
        :param n_trials: number of trials we want to perform
        :param direction: direction to specify if we want to maximize or minimize the value of the metric used
        :param seed: the starting point in generating random numbers
        :param get_hyperparameters_importance: Bool to tell if we want to plot the hyperparameters importance graph
                                                after tuning the hyper parameters
        :param get_parallel_coordinate: Bool to tell if we want to plot the parallel coordinate graph after tuning
        the hyper parameters
        :param get_optimization_history: Bool to tell if we want to plot the optimization history graph
         the hyper parameters
        :param device: "cpu" or "gpu"
        :param recordings_path: the path to the recordings folder where we want to save the data



        """

        # we save the inputs that will be used when tuning the hyper parameters
        self.evaluation_name = evaluation_name
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.sampler = sampler
        self.k = k
        self.l = l
        self.hyper_params = hyper_params
        self.optimization_metric = optimization_metric
        self.evaluation_metrics = evaluation_metrics
        self.direction = direction
        self.seed = seed
        self.get_hyperparameters_importance = get_hyperparameters_importance
        self.get_parallel_coordinate = get_parallel_coordinate
        self.get_optimization_history = get_optimization_history
        self.recordings_path = recordings_path

        assert not (device == 'gpu' and parallelism), "Parallel optimization with gpu is not enabled"

        assert not(path.exists(path.join("Recordings", self.evaluation_name))),\
            "Evaluation with this name already exists"

        self.device = device
        self.parallel = parallelism

    def nested_cross_valid(self, **kwargs):
        """
        Method to call when we want to perform a nested cross validation to evaluate a model

        :return: List containing the scores of the model after performing a nested cross validation
        """

        # we set the seed for the sampling
        if self.seed is not None:
            np_seed(self.seed)

        # We get all the train, test, inner train, qnd inner test sets with our sampler
        all_datasets = self.sampler(k=self.k, l=self.l)

        # we set the seed for the complete nested cross valid operation
        if self.seed is not None:
            manual_seed(self.seed)

        # We create the recording folder and the folder where the recordings of this evaluation will be stored
        makedirs(path.join(self.recordings_path, "Recordings", self.evaluation_name), exist_ok=True)

        # We execute the outter loop in a parallel if we do not train of GPU
        start = time.time()
        subprocess = self.define_subprocess(all_datasets, **kwargs)
        futures = [subprocess.remote(i) for i in range(self.k)]
        scores = ray.get(futures)
        execution_time = time.time() - start
        print(f"Execution time : {execution_time}")

        # We save the evaluation recap
        get_evaluation_recap(evaluation_name=self.evaluation_name, recordings_path=self.recordings_path)

        # We save the hyperparameters plot
        plot_hyperparameter_importance_chart(evaluation_name=self.evaluation_name, recordings_path=self.recordings_path)

        return scores

    def define_subprocess(self, all_datasets, **kwargs):
        if self.device == "cpu" and self.parallel:
            ray.init()
            verbose = False
        else:
            ray.init(num_cpus=1)
            verbose = True

        @ray.remote(num_cpus=1)
        def subprocess(k):
            """
            Executes one fold of the outter cross valid
            :param k: fold iteration
            :return: score
            """
            # We get the train, test and valid sets
            train_set, test_set, valid_set = self.get_datasets(all_datasets[k])

            # We create the Recorder object to save the result of this experience
            recorder = self.create_recorder(index=k)

            # We record the data count
            recorder.record_data_info("train_set", len(train_set))
            recorder.record_data_info("valid_set", len(valid_set))
            recorder.record_data_info("test_set", len(test_set))

            # We create the tuner to perform the hyperparameters optimization
            print(f"Hyperparameter tuning started - K = {k}")
            tuner = self.create_tuner(datasets=all_datasets[k]["inner"],
                                      index=k, **kwargs)

            # We perform the hyper parameters tuning to get the best hyper parameters
            best_hyper_params, hyper_params_importance = tuner.tune(verbose=False)
            print(f"Hyperparameter tuning done - K = {k}")

            # We save the hyperparameters
            recorder.record_hyperparameters(best_hyper_params)

            # We save the hyperparameters importance
            recorder.record_hyperparameters_importance(hyper_params_importance)

            # We create our model with the best hyper parameters
            model = self.create_model(best_hyper_params=best_hyper_params)

            # We create a trainer to train the model
            trainer = self.create_trainer(model=model, best_hyper_params=best_hyper_params, device=self.device)

            # We train our model with the best hyper parameters
            print(f"Final model training - K = {k}")
            trainer.fit(train_set=train_set, val_set=valid_set, verbose=verbose)

            # We save the trained model
            recorder.record_model(model=model)

            # We extract x_cont, x_cat and target from the test set
            x_cont, x_cat, target = self.extract_data(test_set)

            # We get the predictions
            predictions = trainer.predict(x_cont, x_cat)

            # We save the predictions
            recorder.record_predictions(predictions)

            for metric_name, f in self.evaluation_metrics.items():
                # We save the scores, (TO BE UPDATED)
                recorder.record_scores(score=f(predictions, target),
                                       metric=metric_name)
            # We get the score
            score = self.optimization_metric(predictions, target)

            # We save all the data collected in a file
            recorder.generate_file()

            # We calculate the score with the help of the metric function
            return score

        return subprocess

    @staticmethod
    def extract_data(dataset):
        """
        Method to extract the continuous data, categorical data, and the target

        :param dataset: PetaleDataset or PetaleDataframe containing the data

        :return: Python tuple containing the continuous data, categorical data, and the target
        """
        x_cont = dataset.X_cont
        target = dataset.y
        if dataset.X_cat is not None:
            x_cat = dataset.X_cat
        else:
            x_cat = None

        return x_cont, x_cat, target

    @staticmethod
    def get_datasets(dataset_dictionary):
        """
        Method to extract the train, test, and valid sets

        :param dataset_dictionary: Python dictionary that contains the three sets


        :return: Python tuple containing the train, test, and valid sets
        """
        return dataset_dictionary["train"], dataset_dictionary["test"], dataset_dictionary["valid"]

    def create_tuner(self, datasets, index, **kwargs):
        raise NotImplementedError

    def create_model(self, best_hyper_params):
        raise NotImplementedError

    def create_trainer(self, model, **kwargs):
        raise NotImplementedError

    def create_recorder(self, model, **kwargs):
        raise NotImplementedError


class NNEvaluator(Evaluator):

    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, optimization_metric,
                 evaluation_metrics, k, l=1, max_epochs=100,
                 direction="minimize", seed=None, get_hyperparameters_importance=False, get_parallel_coordinate=False,
                 get_optimization_history=False, device="cpu", parallelism=True, early_stopping_activated=False,
                 recordings_path=""):
        """
        Class that will be responsible of the evaluation of the Neural Networks models

        :param max_epochs: the maximum number of epochs to do in training

        """
        super().__init__(model_generator=model_generator, sampler=sampler, hyper_params=hyper_params, n_trials=n_trials,
                         optimization_metric=optimization_metric, evaluation_metrics=evaluation_metrics, k=k, l=l,
                         direction=direction, seed=seed,
                         get_hyperparameters_importance=get_hyperparameters_importance,
                         get_parallel_coordinate=get_parallel_coordinate,
                         get_optimization_history=get_optimization_history,
                         evaluation_name=evaluation_name, device=device, parallelism=parallelism,
                         recordings_path=recordings_path)

        self.max_epochs = max_epochs
        self.early_stopping_activated = early_stopping_activated

    def nested_cross_valid(self, **kwargs):

        # We create the checkpoints folder where the early stopper will save the models
        if self.early_stopping_activated and not path.exists(path.join("checkpoints")):
            mkdir(path.join("checkpoints"))

        scores = super().nested_cross_valid(**kwargs)

        # We delete the files created to save the checkpoints of our model by the early stopper
        if path.exists(path.join("checkpoints")):
            rmtree(path.join("checkpoints"))

        return scores

    def create_tuner(self, datasets, index, **kwargs):
        """
        Method to create the Tuner object that will be used in the hyper parameters tuning

        :param datasets: Python list that contains all the inner train, inner test, amd inner valid sets
        :param index: The index of the split

        """

        return NNTuner(model_generator=self.model_generator, datasets=datasets,
                       hyper_params=self.hyper_params, n_trials=self.n_trials,
                       metric=self.optimization_metric, direction=self.direction, k=self.l,
                       max_epochs=self.max_epochs, study_name=f"{self.evaluation_name}_{index}",
                       get_parallel_coordinate=self.get_parallel_coordinate,
                       get_hyperparameters_importance=self.get_hyperparameters_importance,
                       get_optimization_history=self.get_optimization_history,
                       early_stopping_activated=self.early_stopping_activated,
                       path=path.join(self.recordings_path, "Recordings", self.evaluation_name, f"Split_{index}"), **kwargs)

    def create_model(self, best_hyper_params):
        """
        Method to create the Model

        :param best_hyper_params: Python list that contains a set of hyper parameter used in the creation of the neural
         network model
        """
        return self.model_generator(layers=best_hyper_params[LAYERS], dropout=best_hyper_params[DROPOUT],
                                    activation=best_hyper_params[ACTIVATION])

    def create_trainer(self, model, **kwargs):
        """
        Method to create a trainer object that will be used to train of our model

        :param model: The Neural Network model we want to train
        """
        assert 'best_hyper_params' in kwargs.keys(), 'best_hyper_params argument missing'

        best_hyper_params = kwargs['best_hyper_params']

        return NNTrainer(model, epochs=self.max_epochs, batch_size=best_hyper_params[BATCH_SIZE],
                         lr=best_hyper_params[LR], weight_decay=best_hyper_params[WEIGHT_DECAY],
                         metric=self.optimization_metric, device=kwargs.get('device', 'cpu'),
                         early_stopping_activated=self.early_stopping_activated)

    def create_recorder(self, index):
        """
        Method to create a Recorder to save data about experiments

        :param index: The index of the split
        """
        return NNRecorder(evaluation_name=self.evaluation_name, index=index, recordings_path=self.recordings_path)


class RFEvaluator(Evaluator):

    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, optimization_metric,
                 evaluation_metrics, k, l=1, direction="minimize", seed=None, get_hyperparameters_importance=False,
                 get_parallel_coordinate=False, get_optimization_history=False, recording_path=""):
        """
        Class that will be responsible of the evaluation of the Random Forest models

        """

        super().__init__(model_generator=model_generator, sampler=sampler, hyper_params=hyper_params, n_trials=n_trials,
                         optimization_metric=optimization_metric, evaluation_metrics=evaluation_metrics, k=k, l=l,
                         direction=direction, seed=seed, get_parallel_coordinate=get_parallel_coordinate,
                         get_hyperparameters_importance=get_hyperparameters_importance,
                         get_optimization_history=get_optimization_history,
                         evaluation_name=evaluation_name, device="cpu", parallelism=True,
                         recordings_path=recording_path)

    def create_tuner(self, datasets, index, **kwargs):
        """
        Method to create the Tuner object that will be used in the hyper parameters tuning

        :param datasets: Python list that contains all the inner train, inner test, amd inner valid sets
        :param index: The index of the split



        """
        return RFTuner(study_name=f"{self.evaluation_name}_{index}", model_generator=self.model_generator,
                       datasets=datasets,hyper_params=self.hyper_params, n_trials=self.n_trials,
                       metric=self.optimization_metric, direction=self.direction, k=self.l,
                       get_hyperparameters_importance=self.get_hyperparameters_importance,
                       get_parallel_coordinate=self.get_parallel_coordinate,
                       get_optimization_history=self.get_optimization_history,
                       path=path.join(self.recordings_path, "Recordings", self.evaluation_name, f"Split_{index}"), **kwargs
                       )

    def create_model(self, best_hyper_params):
        """git
        Method to create the Model

        :param best_hyper_params: Python list that contains a set of hyper parameter used in the creation of the Random
         Forest model

        """
        return self.model_generator(n_estimators=best_hyper_params[N_ESTIMATORS],
                                    max_features=best_hyper_params[MAX_FEATURES],
                                    max_depth=best_hyper_params[MAX_DEPTH], max_samples=best_hyper_params[MAX_SAMPLES])

    def create_trainer(self, model, **kwargs):
        """
        Method to create a trainer object that will be used to train of our model

        :param model: The Random Forest model we want to train

        """

        return RFTrainer(model=model, metric=self.optimization_metric)

    def create_recorder(self, index):
        """
        Method to create a Recorder to save data about experiments

        :param index: The index of the split
        """
        return Recorder(evaluation_name=self.evaluation_name, index=index, recordings_path=self.recordings_path)
