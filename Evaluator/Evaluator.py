"""
Authors : Mehdi Mitiche

File that contains the class related to the evaluation of the models

"""
from Training.Training import NNTrainer, RFTrainer
from Tuner.Tuner import NNTuner, RFTuner


class Evaluator:
    def __init__(self, model_generator, sampler, hyper_params, n_trials, metric, k, l=1,
                 direction="minimize", seed=None):
        """
        Class that will be responsible of the evaluation of the model

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


        """

        # we save the inputs that will be used when tuning the hyper parameters
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.sampler = sampler
        self.k = k
        self.l = l
        self.hyper_params = hyper_params
        self.metric = metric
        self.direction = direction
        self.seed = seed

    def nested_cross_valid(self):
        """
        Method to call when we want to perform a nested cross validation to evaluate a model

        :return: the scores of the model after performing a nested cross validation
        """

        # We get all the train, test, inner train, qnd inner test sets with our sampler
        all_datasets = self.sampler(k=self.k, l=self.l)

        # We init the list that will contains the scores
        scores = []

        for i in range(self.k):

            # We the get the train and the test datasets
            train_set, test_set, valid_set = self.get_datasets(all_datasets[i])

            # We create the tuner to perform the hyperparameters optimization
            tuner = self.create_tuner(datasets=all_datasets[i]["inner"])

            # We perform the hyper parameters tuning to get the best hyper parameters
            best_hyper_params = tuner.tune()

            # We create our model with the best hyper parameters
            model = self.create_model(best_hyper_params=best_hyper_params)

            # We create a trainer to train the model
            trainer = self.create_trainer(model=model, best_hyper_params=best_hyper_params)

            # We train our model with the best hyper parameters
            trainer.fit(train_set=train_set, val_set=valid_set)

            # We extract x_cont, x_cat and target from the validset
            x_cont = test_set.X_cont
            target = test_set.y
            if test_set.X_cat is not None:
                x_cat = test_set.X_cat
            else:
                x_cat = None

            # we calculate the score with the help of the metric function
            scores.append(self.metric(trainer.predict(x_cont, x_cat), target))

        return sum(scores) / len(scores)


class NNEvaluator(Evaluator):
    def __init__(self, model_generator, sampler, hyper_params, n_trials, metric, k, l=1, max_epochs=100,
                 direction="minimize", seed=None):
        """
        Class that will be responsible of the evaluation of the Neural Networks models

        :param max_epochs: the maximum number of epochs to do in training

        """
        super().__init__(model_generator=model_generator, sampler=sampler, hyper_params=hyper_params, n_trials=n_trials,
                         metric=metric, k=k, l=l, direction=direction, seed=seed)

        self.max_epochs = max_epochs

    def get_datasets(self, dataset_dictionary):
        """
        Method to extract the train, test, and valid sets

        :param dataset_dictionary: Python dictionary that contains the three sets

        """
        return dataset_dictionary["train"], dataset_dictionary["test"], dataset_dictionary["valid"]

    def create_tuner(self, datasets):
        """
        Method to create the Tuner object that will be used in the hyper parameters tuning

        :param datasets: Python list that contains all the inner train, inner test, amd inner valid sets

        """

        return NNTuner(model_generator=self.model_generator, datasets=datasets,
                       hyper_params=self.hyper_params, n_trials=self.n_trials,
                       metric=self.metric, direction=self.direction, k=self.l, seed=self.seed,
                       max_epochs=self.max_epochs)

    def create_model(self, best_hyper_params):
        """
        Method to create the Model

        :param best_hyper_params: Python list that contains a set of hyper parameter used in the creation of the neural
         network model
        """
        return self.model_generator(layers=best_hyper_params["layers"], dropout=best_hyper_params["dropout"],
                                    activation=best_hyper_params["activation"])

    def create_trainer(self, model, best_hyper_params):
        """
        Method to create a trainer object that will be used to train of our model

        :param model: The Neural Network model we want to train
        :param best_hyper_params: Python list that contains a set of hyper parameter used in the training of the model

        """
        return NNTrainer(model, epochs=self.max_epochs, batch_size=best_hyper_params["batch_size"],
                         lr=best_hyper_params["lr"], weight_decay=best_hyper_params["weight_decay"],
                         seed=self.seed, metric=self.metric)


class RFEvaluator(Evaluator):
    def __init__(self, model_generator, sampler, hyper_params, n_trials, metric, k, l=1, max_epochs=100,
                 direction="minimize", seed=None):
        """
        Class that will be responsible of the evaluation of the Random Forest models

        """

        super().__init__(model_generator=model_generator, sampler=sampler, hyper_params=hyper_params, n_trials=n_trials,
                         metric=metric, k=k, l=l, direction=direction, seed=seed)

    def get_datasets(self, dataset_dictionary):
        """
        Method to extract the train, test, and valid sets

        :param dataset_dictionary: Python dictionary that contains the three sets

        """
        return dataset_dictionary["train"], dataset_dictionary["test"], None

    def create_tuner(self, datasets):
        """
        Method to create the Tuner object that will be used in the hyper parameters tuning

        :param datasets: Python list that contains all the inner train, inner test, amd inner valid sets

        """
        return RFTuner(model_generator=self.model_generator, datasets=datasets,
                       hyper_params=self.hyper_params, n_trials=self.n_trials,
                       metric=self.metric, direction=self.direction, k=self.l, seed=self.seed)

    def create_model(self, best_hyper_params):
        """
        Method to create the Model

        :param best_hyper_params: Python list that contains a set of hyper parameter used in the creation of the Random
         Forest model

        """
        return self.model_generator(n_estimators=best_hyper_params["n_estimators"])

    def create_trainer(self, model, best_hyper_params):
        """
        Method to create a trainer object that will be used to train of our model

        :param model: The Random Forest model we want to train
        :param best_hyper_params: Python list that contains a set of hyper parameter used in the training of the model

        """

        return RFTrainer(model)
