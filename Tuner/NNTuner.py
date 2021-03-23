"""
Authors : Mehdi Mitiche

Files that contains the logic related to hyper parameters tuning

"""
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna.visualization import plot_param_importances

from Training.Training import Trainer
from Hyperparameters.constants import *


class Objective:
    def __init__(self, model_generator, datasets, hyper_params, k, metric, max_epochs):
        """
        Method that will fit the model to the given data
        
        :param model_generator: Instance of the ModelGenerator class that will be responsible of generating the model
        :param datasets: Datasets representing all the train and test sets to be used in the cross validation
        :param hyper_params: Dictionary containing information of the hyper parameter we want to tune
        :param k: Dumber of folds to use in the cross validation
        :param metric: Function that takes the output of the model and the target
                       and returns the metric we want to optimize

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

        # We create the Trainer that will train our model
        trainer = Trainer(model)

        # We perform a k fold cross validation to evaluate the model
        score = trainer.cross_valid(datasets=self.datasets, batch_size=batch_size, lr=lr,
                                    epochs=self.max_epochs, metric=self.metric, k=self.k,
                                    weight_decay=weight_decay, trial=trial)

        # we return the score
        return score


class NNTuner:
    def __init__(self, model_generator, datasets, hyper_params, k, n_trials, metric, study_name, max_epochs=100,
                 direction="minimize", **kwargs):
        """
        Class that will be responsible of the hyperparameters tuning
        
        :param model_generator: Instance of the ModelGenerator class that will be responsible of generating the model
        :param datasets: PetaleDatasets representing all the train and test sets to be used in the cross validation
        :param hyper_params: Dictionary containing information of the hyper parameter
                             we want to tune (min, max, step, values)
        :param k: Number of folds to use in the cross validation
        :param metric: Function that takes the output of the model and the target and returns
                       the metric we want to optimize
        :param study_name: String that represents the name of the study
        :param n_trials: Number of trials we want to perform
        :param direction: Direction to specify if we want to maximize or minimize the value of the metric used

        """
        # We look for keyword args
        n_startup = kwargs.get('n_startup_trial', 10)
        n_ei_candidates = kwargs.get('n_ei_candidates', 20)
        min_resource = kwargs.get('min_resource', 10)
        eta = kwargs.get('eta', 4)

        # we create the study 
        self.study = create_study(direction=direction, study_name=study_name,
                                  sampler=TPESampler(n_startup_trials=n_startup, n_ei_candidates=n_ei_candidates),
                                  pruner=SuccessiveHalvingPruner(min_resource=min_resource, reduction_factor=eta))

        # we save the inputs that will be used when tuning the hyper parameters
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.datasets = datasets
        self.hyper_params = hyper_params
        self.k = k
        self.metric = metric
        self.max_epochs = max_epochs

    def tune(self):
        """
        Method to call to tune the hyperparameters of a given model
        
        :return: the result of the study containing the best trial and the best values of each hyper parameter
        """
        # We perform the optimization
        self.study.optimize(
            Objective(model_generator=self.model_generator, datasets=self.datasets,
                      hyper_params=self.hyper_params, k=self.k, metric=self.metric,
                      max_epochs=self.max_epochs), self.n_trials)

        # We extract the best trial
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

    def get_hyper_params_importance(self):
        """
        Method to plot the hyper parameters graph and save it in a html file
        """

        # We generate the hyper parameters importance graph with optuna
        fig = plot_param_importances(self.study)

        # We save the graph in a html file to have an interactive graph
        fig.write_html(f"../FeatureImportance/{self.study.study_name}")
