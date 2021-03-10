"""
Authors : Mehdi Mitiche

File that contains the class related to the evaluation of the models

"""
from Tuner.NNTuner import NNTuner

class NNEvaluator:
    def __init__(self, model_generator, learningset, hyper_params, n_trials, metric, k, l, max_epochs = 100, direction="minimize"):
        """
        Class that will be responsible of the hyperparameters tuning
        
        :param model_generator: instance of the ModelGenerator class that will be responsible of generating the model
        :param learningset: Petale Dataset containing the learning set
        :param hyper_params: dictionary containg information of the hyper parameter we want to tune : min, max, step, values
        :param metric: a function that takes the output of the model and the target and returns  the metric we want to optimize
        :param k: Number of folds in the outer cross validation
        :param l: Number of folds in the inner cross validation
        :param n_trials: number of trials we want to perform
        :param max_epochs:the maximal number of epochs to do in the training
        :param direction: direction to specify if we want to maximize or minimize the value of the metric used

        """

        # we save the inputs that will be used when tuning the hyoer parameters
        self.n_trials = n_trials
        self.model_generator = model_generator
        self.learningset = learningset
        self.hyper_params = hyper_params
        self.metric = metric
        self.max_epochs = max_epochs
        self.direction = direction
    def nested_cross_valid(self):
        """
        Method to call when we want to perform a nested cross validation and evaluate the model
        
        :return: the score of the model after peroforming a nested cross calidation
        """

        #we init the list that will contains the scores
        scores = []


        return scores