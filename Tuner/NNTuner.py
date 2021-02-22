"""
Authors : Mehdi Mitiche

Files that contains the logic related to hyper parameters tuning

"""
from optuna import create_study
from Training.Training import Trainer

class objective():
    def __init__(self, model, dataset, hyper_params, metric ):
        """
        Method that will fit the model to the given data
        
        :param model: class of the model we want to use 
        :param dataset: Petale Dataset containing the training set
        :param hyper_params: dictionary containg information of the hyper parameter we want to tune : min, max, step, values
        :param metric: type of the metric we want to optimize


        :return: the value of the metric after performing a k fold cross validation on the model with a subset of the given hyper parameter
        """
        # we save the inputs that will be used when calling the class
        self.model = model
        self.dataset = dataset
        self.hyper_params = hyper_params
        self.metric = metric
    def __call__(self, trial):
        hyper_params = self.hyper_params
        # We optimize the number of hidden layers
        n_layers = trial.suggest_int("n_layers",hyper_params["n_layers"]["min"], hyper_params["n_layers"]["max"])
        layers= []
        # We optimize the number of node in each hidden layer
        for i in range(n_layers):
            out_features = trial.suggest_int(f"n_units{i}",hyper_params["n_units"]["min"], hyper_params["n_units"]["max"])
            layers.append(out_features)
        # We optimize the dropout probability
        p = trial.suggest_uniform("dropout",hyper_params["dropout"]["min"], hyper_params["dropout"]["max"])
        # We optimize the batch size used in the training
        batch_size = trial.suggest_int("batch_size",hyper_params["batch_size"]["min"], hyper_params["batch_size"]["max"])
        # We optimize the optimizer that will be used in the training
        optimizer_name = trial.suggest_categorical("optimizer_name", hyper_params["optimizer_name"]["values"])
        # We optimize the value of the learning rate
        lr = trial.suggest_loguniform("lr",hyper_params["lr"]["min"], hyper_params["lr"]["max"])

        if(self.dataset.X_cat is None):
            cat_sizes = None
        else:
            #this case will be implemented very soon
            cat_sizes = None

        
        # We define the model with the suggested set of hyper parameters
        model = self.model(self.dataset.X_cont.shape[1],layers = layers, dropout= p, cat_sizes=cat_sizes)

        #we creat the Trainer that will train our model
        trainer = Trainer(model)
        #we perform a k fold cross validation to evaluate the model
        score = trainer.cross_valid(self.dataset, batch_size=batch_size, optimizer_name=optimizer_name,lr=lr,epochs=200)

        #we return the score 
        return score


class NNTuner:
    def __init__(self, model, dataset, hyper_params, n_trials, metric="loss", direction="minimize"):
        """
        Class that will be responsible of the hyperparameters tuning
        
        :param model: class of the model we want to use 
        :param dataset: Petale Dataset containing the training set
        :param hyper_params: dictionary containg information of the hyper parameter we want to tune : min, max, step, values
        :param metric: type of the metric we want to optimize
        :param n_trials: number of trials we want to perform
        :param direction: direction to specify if we want to maximize or minimize the value of the metric used

        """
        # we create the study 
        self.study = create_study() 

        # we save the inputs that will be used when tuning the hyoer parameters
        self.n_trials = n_trials
        self.model = model
        self.dataset = dataset
        self.hyper_params = hyper_params
        self.metric = metric
        self.direction = direction
    def tune(self):
        """
        Method to call when we want to tune the hyperparameters
        
        :return: the result of the study containg the best trial and the best values of each hyper parameter
        """
        # we perform the optimization 
        self.study.optimize(objective(self.model, self.dataset, self.hyper_params, self.metric ),self.n_trials)  
        return self.study 
