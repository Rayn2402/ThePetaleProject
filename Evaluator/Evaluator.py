"""
Authors : Mehdi Mitiche

File that contains the class related to the evaluation of the models

"""
from Training.Training import NNTrainer, RFTrainer
from Tuner.Tuner import NNTuner, RFTuner
from torch import manual_seed
from numpy.random import seed as np_seed
from Hyperparameters.constants import *
from Recorder.Recorder import Recorder
from torch.nn import Softmax
from torch import unique, argmax




class Evaluator:
    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, metric, k, l=1,
                 direction="minimize", seed=None, plot_feature_importance=False, plot_intermediate_values=False):
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

        for i in range(self.k):
            # We get the train, test and valid sets
            train_set, test_set, valid_set = self.get_datasets(all_datasets[i])

            # We create the Recorder object to save the result of this experience
            recorder = Recorder(evaluation_name=self.evaluation_name, index=i)

            # We create the tuner to perform the hyperparameters optimization
            tuner = self.create_tuner(datasets=all_datasets[i]["inner"], study_name=f"{self.evaluation_name}_{i}",
                                      **kwargs)

            # We perform the hyper parameters tuning to get the best hyper parameters
            best_hyper_params = tuner.tune()

            # We save the hyperparameters
            recorder.record_hyperparameters(best_hyper_params)

            # We create our model with the best hyper parameters
            model = self.create_model(best_hyper_params=best_hyper_params)

            # We create a trainer to train the model
            trainer = self.create_trainer(model=model, best_hyper_params=best_hyper_params)

            # We train our model with the best hyper parameters
            trainer.fit(train_set=train_set, val_set=valid_set)

            # We save the trained model
            recorder.record_model(model=model)

            # We extract x_cont, x_cat and target from the test set
            x_cont, x_cat, target = self.extract_data(test_set)

            # We get the predictions
            predictions = trainer.predict(x_cont, x_cat)
            print(argmax(predictions, dim=1))

            # We initialize the Softmax object
            softmax = Softmax(dim=1)

            # We save the predictions
            recorder.record_predictions(softmax(predictions))



            # We get the score
            score = self.metric(predictions, target)

            # We save the scores, (TO BE UPDATED)
            recorder.record_scores(score=score, metric="ACCURACY")

            # We calculate the score with the help of the metric function
            scores.append(self.metric(trainer.predict(x_cont, x_cat), target))

            # We save all the data collected in a file
            recorder.generate_file()

        return scores

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


class NNEvaluator(Evaluator):
    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, metric, k, l=1, max_epochs=100,
                 direction="minimize", seed=None, plot_feature_importance=False, plot_intermediate_values=False):
        """ sets
 that con
        Class that will be responsible of the evaluation of the Neural Networks models

        :param max_epochs: the maximum number of epochs to do in training

        """
        super().__init__(model_generator=model_generator, sampler=sampler, hyper_params=hyper_params, n_trials=n_trials,
                         metric=metric, k=k, l=l, direction=direction, seed=seed,
                         plot_feature_importance=plot_feature_importance,
                         plot_intermediate_values=plot_intermediate_values, evaluation_name=evaluation_name)

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

    def create_trainer(self, model, best_hyper_params):
        """
        Method to create a trainer object that will be used to train of our model

        :param model: The Neural Network model we want to train
        :param best_hyper_params: Python list that contains a set of hyper parameter used in the training of the model

        """
        return NNTrainer(model, epochs=self.max_epochs, batch_size=best_hyper_params[BATCH_SIZE],
                         lr=best_hyper_params[LR], weight_decay=best_hyper_params[WEIGHT_DECAY],
                         metric=None)


class RFEvaluator(Evaluator):
    def __init__(self, evaluation_name, model_generator, sampler, hyper_params, n_trials, metric, k, l=1, max_epochs=100,
                 direction="minimize", seed=None, plot_feature_importance=False, plot_intermediate_values=False):
        """
        Class that will be responsible of the evaluation of the Random Forest models

        """

        super().__init__(model_generator=model_generator, sampler=sampler, hyper_params=hyper_params, n_trials=n_trials,
                         metric=metric, k=k, l=l, direction=direction, seed=seed,
                         plot_intermediate_values=self.plot_intermediate_values,
                         plot_feature_importance=self.plot_feature_importance, evaluation_name=evaluation_name)

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

    def create_trainer(self, model, best_hyper_params):
        """
        Method to create a trainer object that will be used to train of our model

        :param model: The Random Forest model we want to train
        :param best_hyper_params: Python list that contains a set of hyper parameter used in the training of the model

        """

        return RFTrainer(model=model, metric=self.metric)
