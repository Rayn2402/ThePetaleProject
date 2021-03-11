"""
Authors : Mehdi Mitiche

File that contains the class related to the evaluation of the models

"""
from Training.Training import Trainer
from Tuner.NNTuner import NNTuner


class NNEvaluator:
    def __init__(self, model_generator, sampler, hyper_params, n_trials, metric, k, l, max_epochs=100,
                 direction="minimize"):
        """
        Class that will be responsible of the evolution of the model
        
        :param model_generator: instance of the ModelGenerator class that will be responsible of generating the model
        :param sampler: A sampler object that will be called to perform the stratified sampling to get all the train
        and test set for both the inner and the outer training
        :param hyper_params: dictionary containing information of the hyper parameter we want to tune
        :param metric: a function that takes the output of the model and the target and returns  the metric we want
        to optimize
        :param k: Number of folds in the outer cross validation
        :param l: Number of folds in the inner cross validation
        :param n_trials: number of trials we want to perform
        :param max_epochs:the maximal number of epochs to do in the training
        :param direction: direction to specify if we want to maximize or minimize the value of the metric used

        """

        # we save the inputs that will be used when tuning the hyper parameters
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.sampler = sampler
        self.k = k
        self.l = l
        self.hyper_params = hyper_params
        self.metric = metric
        self.max_epochs = max_epochs
        self.direction = direction

    def nested_cross_valid(self):
        """
        Method to call when we want to perform a nested cross validation and evaluate the model
        
        :return: the scores of the model after performing a nested cross validation
        """

        # we get all the train, test, inner train, qnd inner test sets with our sampler
        all_datasets = self.sampler(k=self.k, l=self.l)

        # we init the list that will contains the scores
        scores = []

        for i in range(self.k):
            # we the get the train and the test datasets
            train_set, valid_set, test_set = all_datasets[i]["train"], all_datasets[i]["valid"], all_datasets[i]["test"]

            # we create the tuner to perform the hyperparameters optimisation
            tuner = NNTuner(model_generator=self.model_generator, datasets=all_datasets[i]["inner"],
                            hyper_params=self.hyper_params, n_trials=self.n_trials,
                            metric=self.metric, direction=self.direction, k=self.l)

            # we perform the hyper parameters tuning to get the best hyper parameters
            best_hyper_params = tuner.tune()

            # we create our model with the best hyper parameters
            model = self.model_generator(layers=best_hyper_params["layers"], dropout=best_hyper_params["dropout"])

            # we create a trainer to train the model
            trainer = Trainer(model)
            # we train our model with the best hyper parameters
            trainer.fit(train_set=train_set, val_set=valid_set, epochs=self.max_epochs,
                        batch_size=best_hyper_params["batch_size"],
                        lr=best_hyper_params["lr"], weight_decay=best_hyper_params["weight_decay"])

            # we extract x_cont, x_cat and target from the validset
            x_cont = test_set.X_cont
            target = test_set.y
            if test_set.X_cat is not None:
                x_cat = test_set.X_cat
            else:
                x_cat = None

            # we calculate the score with the help of the metric function
            scores.append(self.metric(model(x_cont, x_cat).float(), target))

        return sum(scores) / len(scores)
