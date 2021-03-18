"""
Authors : Mehdi Mitiche

Files that contains the logic related to hyper parameters tuning

"""
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.visualization import plot_param_importances

from Training.Training import NNTrainer, RFTrainer
from Hyperparameters.constants import *


class NNObjective:
    def __init__(self, model_generator, datasets, hyper_params, k, metric, max_epochs, seed=None):
        """
        Class that will represent the objective function for tuning Neural networks
        
        :param model_generator: Instance of the ModelGenerator class that will be responsible of generating the model
        :param datasets: Datasets representing all the train and test sets to be used in the cross validation
        :param hyper_params: Dictionary containing information of the hyper parameter we want to tune
        :param k: Dumber of folds to use in the cross validation
        :param metric: Function that takes the output of the model and the target
                       and returns the metric we want to optimize
        :param seed: the starting point in generating random numbers

        :return: the value of the metric after performing a k-fold cross validation
                 on the model with a subset of the given hyper parameter
        """

        # we save the inputs that will be used when calling the class
        self.model_generator = model_generator
        self.datasets = datasets
        self.hyper_params = hyper_params
        self.k = k
        self.metric = metric
        self.max_epochs = max_epochs
        self.seed = seed

    def __call__(self, trial):
        hyper_params = self.hyper_params

        # We sample a number of hidden layers
        n_layers = trial.suggest_int(N_LAYERS,
                                     hyper_params[N_LAYERS][MIN],
                                     hyper_params[N_LAYERS][MAX])

        # We sample a number of node in each hidden layer
        layers = [trial.suggest_int(f"{N_UNITS}{i}", hyper_params[N_UNITS][MIN],
                                    hyper_params[N_UNITS][MAX]) for i in range(n_layers)]

        # We sample the dropout probability
        p = trial.suggest_uniform(DROPOUT,
                                  hyper_params[DROPOUT][MIN],
                                  hyper_params[DROPOUT][MAX])

        # We sample a value for the batch size used in the training
        batch_size = trial.suggest_int(BATCH_SIZE,
                                       hyper_params[BATCH_SIZE][MIN],
                                       hyper_params[BATCH_SIZE][MAX])

        # We sample a value for the learning rate
        lr = trial.suggest_uniform(LR,
                                   hyper_params[LR][MIN],
                                   hyper_params[LR][MAX])

        # We sample a value for the weight decay used in the training
        weight_decay = trial.suggest_uniform(WEIGHT_DECAY,
                                             hyper_params[WEIGHT_DECAY][MIN],
                                             hyper_params[WEIGHT_DECAY][MAX])

        # We sample a type of activation function for the network
        activation = trial.suggest_categorical(ACTIVATION,
                                               hyper_params[ACTIVATION][VALUES])

        # We define the model with the suggested set of hyper parameters
        model = self.model_generator(layers=layers, dropout=p, activation=activation)

        # we create the Trainer that will train our model
        trainer = NNTrainer(model=model, batch_size=batch_size, lr=lr, epochs=self.max_epochs,
                            weight_decay=weight_decay)
        # we perform a k fold cross validation to evaluate the model
        score = trainer.cross_valid(datasets=self.datasets, metric=self.metric, k=self.k, trial=trial)

        # we return the score
        return score


class RFObjective:
    def __init__(self, model_generator, datasets, hyper_params, k, metric, max_epochs, seed=None):
        """
        Class that will represent the objective function for tuning Random Forests


        :param model_generator: instance of the ModelGenerator class that will be responsible of generating the model
        Datasets representing all the train and test sets to be used in the cross validation
        :param hyper_params:dictionary containing information of the hyper parameter we want to tune
        :param k:number of folds to use in the cross validation
        :param metric: a function that takes the output of the model and the target and returns the metric we want
        to optimize
        :param seed: the starting point in generating random numbers

        :return: the value of the metric after performing a k fold cross validation on the model with a subset of the
        given hyper parameter
        """

        # we save the inputs that will be used when calling the class
        self.model_generator = model_generator
        self.datasets = datasets
        self.hyper_params = hyper_params
        self.k = k
        self.metric = metric
        self.max_epochs = max_epochs
        self.seed = seed

    def __call__(self, trial):
        hyper_params = self.hyper_params

        # We optimize the the number of estimators used un the training
        n_estimators = trial.suggest_int("n_estimators", hyper_params["n_estimators"]["min"],
                                         hyper_params["n_estimators"]["max"])

        # We define the model with the suggested set of hyper parameters
        model = self.model_generator(n_estimators=n_estimators)

        # we create the Trainer that will train our model
        trainer = RFTrainer(model=model)

        # we perform a k fold cross validation to evaluate the model
        score = trainer.cross_valid(datasets=self.datasets, metric=self.metric, k=self.k, trial=trial)

        # we return the score
        return score


class Tuner:
    def __init__(self, model_generator, datasets, hyper_params, k, n_trials, metric, direction="minimize", seed=None):
        """
                Class that will be responsible of the hyperparameters tuning

                :param model_generator: instance of the ModelGenerator class that will be responsible of generating the model
                :param datasets: Petale Datasets representing all the train and test sets to be used in the cross validation
                :param hyper_params: dictionary containing information of the hyper parameter we want to tune : min, max,
                step, values
                :param k: number of folds to use in the cross validation
                :param metric: a function that takes the output of the model and the target and returns
                the metric we want to optimize
                :param n_trials: number of trials we want to perform
                :param direction: direction to specify if we want to maximize or minimize the value of the metric used
                :param seed: the starting point in generating random numbers

                """
        # we create the study
        self.study = create_study(direction=direction, sampler=TPESampler(n_startup_trials=10, n_ei_candidates=20),
                                  pruner=SuccessiveHalvingPruner(min_resource=5, reduction_factor=4))

        # we save the inputs that will be used when tuning the hyper parameters
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.datasets = datasets
        self.hyper_params = hyper_params
        self.k = k
        self.metric = metric
        self.seed = seed

    def tune(self):
        """
        Method to call to tune the hyperparameters of a given model

        :return: the result of the study containing the best trial and the best values of each hyper parameter
        """
        # We perform the optimization
        self.study.optimize(
            self.Objective(model_generator=self.model_generator, datasets=self.datasets, hyper_params=self.hyper_params,
                           k=self.k, metric=self.metric, max_epochs=self.max_epochs, seed=self.seed), self.n_trials)

        return self.get_best_hyperparams()

    def get_hyper_params_importance(self):
        """
        Method to plot the hyper parameters graph and save it in a ht;l file
        """

        # We generate the hyper parameters importance graph with optuna
        fig = plot_param_importances(self.study)

        # We save the graph in a html file to have an interactive graph
        fig.write_html(f"../FeatureImportance/{self.study.study_name}")


class NNTuner(Tuner):
    def __init__(self, model_generator, datasets, hyper_params, k, n_trials, metric,
                 direction="minimize", seed=None, max_epochs=100):
        """
        Class that will be responsible of tuning Neural Networks

        """
        super().__init__(model_generator, datasets, hyper_params, k, n_trials, metric, direction, seed)
        self.Objective = NNObjective
        self.max_epochs = max_epochs

    def get_best_hyperparams(self):
        # we extract the best trial

        best_trial = self.study.best_trial

        # We extract the best architecture of the model
        n_units = [key for key in best_trial.params.keys() if "n_units" in key]
        if n_units is not None:
            layers = list(map(lambda n_unit: best_trial.params[n_unit], n_units))
        else:
            layers = []

        # We return the best hyperparameters
        return {
            LAYERS: layers,
            DROPOUT: best_trial.params[DROPOUT],
            LR: best_trial.params[LR],
            BATCH_SIZE: best_trial.params[BATCH_SIZE],
            WEIGHT_DECAY: best_trial.params[WEIGHT_DECAY],
            ACTIVATION: best_trial.params[ACTIVATION]
        }


class RFTuner(Tuner):
    def __init__(self, model_generator, datasets, hyper_params, k, n_trials, metric,
                 direction="minimize", seed=None):
        """
        Class that will be responsible of tuning Random Forests

        """
        super().__init__(model_generator, datasets, hyper_params, k, n_trials, metric, direction, seed)
        self.Objective = RFObjective
        self.max_epochs = None

    def get_best_hyperparams(self):
        # we extract the best trial
        best_trial = self.study.best_trial

        # we return the best hyperparameters
        return {
            "n_estimators": best_trial.params["n_estimators"],
        }
