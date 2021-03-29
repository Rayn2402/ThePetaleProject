"""
Authors : Mehdi Mitiche

File that contains the class related to the evaluation of the models

"""
from Training.Training import NNTrainer, RFTrainer
from Tuner.Tuner import NNTuner, RFTuner
from torch import manual_seed
from numpy.random import seed as np_seed
from Hyperparameters.constants import *
from pathos.multiprocessing import ProcessingPool
from pathos.pp import ParallelPool
from multiprocessing import current_process
import os

class Evaluator:
    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, metric, k, l=1,
                 direction="minimize", seed=None, plot_feature_importance=False, plot_intermediate_values=False,
                 device="cpu"):
        """
        Class that will be responsible of the evaluation of the model

        :param evaluation_name: String that represents the name of the evaluation
        :param model_generator: instance of the ModelGenerator class that will be responsible of generating the model
        :param sampler: A sampler object that will be called to perform the stratified sampling to get all the train
        and test set for both the inner and the outer training
        :param hyper_params: dictionary containing information of the hyper parameter we want to tune
        :param metric: a function that takes the output of the model and the target and returns  the metric we want
        to optimize
        :param k: Number of folds in the outer cross validation
        :param l: Number of folds in the inner cross validation
        :param n_trials: number of trials we want to perform
        :param direction: direction to specify if we want to maximize or minimize the value of the metric used
        :param seed: the starting point in generating random numbers
        :param plot_feature_importance: Bool to tell if we want to plot the feature importance graph after tuning
         the hyper parameters
        :param plot_intermediate_values: Bool to tell if we want to plot the intermediate values graph after tuning
         the hyper parameters
        :param device: "cpu" or "gpu"


        """

        # we save the inputs that will be used when tuning the hyper parameters
        self.evaluation_name = evaluation_name
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.sampler = sampler
        self.k = k
        self.l = l
        self.hyper_params = hyper_params
        self.metric = metric
        self.direction = direction
        self.seed = seed
        self.plot_feature_importance = plot_feature_importance
        self.plot_intermediate_values = plot_intermediate_values
        self.device = device

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

        # We init the list that will contain the scores
        scores = []

        subprocess = self.define_subprocess(all_datasets, **kwargs)
        pool = ProcessingPool()
        scores = pool.map(subprocess, range(self.k))
        pool.close()

        # with ProcessPoolExecutor() as executor:
        #     #inputs = list(zip(list(range(self.k)), [all_datasets]*self.k, [kwargs]*self.k))
        #     future = [executor.submit(self.subprocess(i, all_datasets, **kwargs)) for i in range(self.k)]
        #     for fut in as_completed(future):
        #         print(f"The outcome is {fut.result()}")
        #     #scores = executor.map(self.subprocess, inputs)

        return scores

    def define_subprocess(self, all_datasets, **kwargs):

        def subprocess(k):
            """
            Executes one fold of the outter cross valid
            :param k: fold iteration
            :return: score
            """
            # We get the train, test and valid sets
            train_set, test_set, valid_set = self.get_datasets(all_datasets[k])

            # We create the tuner to perform the hyperparameters optimization
            print(f"{k}-th outter loop hyperparameter tuning started - Process ID {current_process().name}")
            tuner = self.create_tuner(datasets=all_datasets[k]["inner"],
                                      study_name=f"{self.evaluation_name}_{k}", **kwargs)

            # We perform the hyper parameters tuning to get the best hyper parameters
            best_hyper_params = tuner.tune()
            print(f"{k}-th outter loop hyperparameter tuning done - Process ID {current_process().name}")

            # We create our model with the best hyper parameters
            model = self.create_model(best_hyper_params=best_hyper_params)

            # We create a trainer to train the model
            trainer = self.create_trainer(model=model, best_hyper_params=best_hyper_params, device=self.device)

            # We train our model with the best hyper parameters
            print(f"{k}-th outter loop final model training - Process ID {current_process().name}")
            trainer.fit(train_set=train_set, val_set=valid_set)

            # We extract x_cont, x_cat and target from the test set
            x_cont, x_cat, target = self.extract_data(test_set)

            # We calculate the score with the help of the metric function
            return self.metric(trainer.predict(x_cont, x_cat), target)

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

    def create_tuner(self, datasets, study_name, **kwargs):
        raise NotImplementedError

    def create_model(self, best_hyper_params):
        raise NotImplementedError

    def create_trainer(self, model, **kwargs):
        raise NotImplementedError


class NNEvaluator(Evaluator):

    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, metric, k, l=1, max_epochs=100,
                 direction="minimize", seed=None, plot_feature_importance=False, plot_intermediate_values=False,
                 device="cpu"):
        """ sets
 that con
        Class that will be responsible of the evaluation of the Neural Networks models

        :param max_epochs: the maximum number of epochs to do in training

        """
        super().__init__(model_generator=model_generator, sampler=sampler, hyper_params=hyper_params, n_trials=n_trials,
                         metric=metric, k=k, l=l, direction=direction, seed=seed,
                         plot_feature_importance=plot_feature_importance,
                         plot_intermediate_values=plot_intermediate_values,
                         evaluation_name=evaluation_name, device=device)

        self.max_epochs = max_epochs

    def create_tuner(self, datasets, study_name, **kwargs):
        """
        Method to create the Tuner object that will be used in the hyper parameters tuning

        :param datasets: Python list that contains all the inner train, inner test, amd inner valid sets
        :param study_name: String that represents the name of the study

        """

        return NNTuner(model_generator=self.model_generator, datasets=datasets,
                       hyper_params=self.hyper_params, n_trials=self.n_trials,
                       metric=self.metric, direction=self.direction, k=self.l,
                       max_epochs=self.max_epochs, study_name=study_name,
                       plot_intermediate_values=self.plot_intermediate_values,
                       plot_feature_importance=self.plot_feature_importance, **kwargs)

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
                         metric=self.metric, device=kwargs.get('device', 'cpu'))


class RFEvaluator(Evaluator):

    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, metric, k, l=1,
                 direction="minimize", seed=None, plot_feature_importance=False, plot_intermediate_values=False):
        """
        Class that will be responsible of the evaluation of the Random Forest models

        """

        super().__init__(model_generator=model_generator, sampler=sampler, hyper_params=hyper_params, n_trials=n_trials,
                         metric=metric, k=k, l=l, direction=direction, seed=seed,
                         plot_intermediate_values=plot_intermediate_values,
                         plot_feature_importance=plot_feature_importance,
                         evaluation_name=evaluation_name, device="cpu")

    def create_tuner(self, datasets, study_name, **kwargs):
        """
        Method to create the Tuner object that will be used in the hyper parameters tuning

        :param datasets: Python list that contains all the inner train, inner test, amd inner valid sets
        :param study_name: String that represents the name of the study


        """
        return RFTuner(study_name=study_name, model_generator=self.model_generator, datasets=datasets,
                       hyper_params=self.hyper_params, n_trials=self.n_trials,
                       metric=self.metric, direction=self.direction, k=self.l,
                       plot_feature_importance=self.plot_feature_importance,
                       plot_intermediate_values=self.plot_intermediate_values, **kwargs
                       )

    def create_model(self, best_hyper_params):
        """
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

        return RFTrainer(model=model, metric=self.metric)
