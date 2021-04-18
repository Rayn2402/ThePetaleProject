"""
Authors : Mitiche

Files that contains class related to the Trainer of the models

"""
import torch
import ray

from Training.EarlyStopping import EarlyStopping
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch import optim, manual_seed, cuda, tensor
from torch import device as device_
from torchcontrib.optim import SWA
from numpy import mean, std


class Trainer:
    def __init__(self, model, metric, device="cpu"):
        """
        Creates a Trainer that will train and evaluate a given model.

        :param model: The model to be trained
        :param device: The device where we want to run our training, this parameter can take two values : "cpu" or "gpu"
        :param metric: Function that takes the output of the model and the target and returns  the metric we want
                       to optimize
        """

        # We save the model in the attribute model
        self.model = model

        # We save the attribute device
        self.device = device_("cuda:0" if cuda.is_available() and device == "gpu" else "cpu")

        # We save the metric
        self.metric = metric

    def inner_random_subsampling(self, datasets, l=5, seed=None):
        """
            Method that will perform a cross validation on the model

            :param datasets: PetaleDatasets representing all the inner train, valid, and test sets
            :param l: Number of random subsampling splits
            :param seed: the starting point in generating random numbers

            :return: The score after performing the cross validation
        """

        # Seed is left to None if fit is called by NNTuner
        if seed is not None:
            manual_seed(seed)

        # We train and test on each of the inner split
        subprocess = self.define_subprocess(datasets)
        futures = [subprocess.remote(i) for i in range(l)]
        scores = ray.get(futures)

        # We the mean of the scores divided by the standard deviation
        standard_dev = 1 if len(scores) == 1 else std(scores)
        return mean(scores) / standard_dev

    def define_subprocess(self, datasets):
        """
        Builds the subprocess according to the device

        :param datasets: PetaleDatasets representing all the inner train, valid, and test sets
        :return: subprocess function
        """

        gpus = 1 if (self.device == "gpu") else 0

        @ray.remote(num_gpus=gpus)
        def subprocess(i):
            """
            Consists of the parallelizable process of the inner random subsampling

            :param i: i-th inner random subsamples' splits on which to test hyperparameters selection
            """
            # We the get the train, test, valid sets of the step we are currently in
            train_set, test_set, valid_set = self.get_datasets(datasets[i])

            # we train our model with the train and valid sets
            self.fit(train_set=train_set, val_set=valid_set)

            # We extract x_cont, x_cat and target from the test set
            x_cont, x_cat, target = self.extract_data(test_set)

            # We calculate the score with the help of the metric function
            score = self.metric(self.predict(x_cont=x_cont, x_cat=x_cat, log_prob=True), target)

            # We save the score
            return score

        return subprocess

    def extract_batch(self, batch_list):
        """
        Extracts the continuous data (X_cont), the categorical data (X_cat) and the ground truth (y)

        :param batch_list: list containing a batch from dataloader
        :return: 3 tensors
        """

        if len(batch_list) > 2:
            x_cont, x_cat, y = batch_list
            x_cont, x_cat, y = x_cont.to(self.device), x_cat.to(self.device), y.to(self.device)
        else:
            x_cont, y = batch_list
            x_cont, y = x_cont.to(self.device), y.to(self.device)
            x_cat = None

        return x_cont, x_cat, y

    def update_trainer(self, **kwargs):
        """
        Abstract method to update trainer internal attributes
        """
        raise NotImplementedError

    def fit(self, train_set, val_set):
        """
        Abstract methods to train the model

        :param train_set: Training set
        :param val_set: Validation set
        """
        raise NotImplementedError

    def predict(self, x_cont, x_cat, **kwargs):

        """
        Abstract methods that return prediction of a model
        (log probabilities in case of classification and real-valued number in case of regression)

        :param x_cont: tensor with continuous inputs
        :param x_cat: tensor with categorical ordinal encoding
        """
        raise NotImplementedError

    def extract_data(self, dataset):
        """
        Abstract method to extract data from datasets

        :param dataset: PetaleDataset or PetaleDataframe containing the data
        :return: Python tuple containing the continuous data, categorical data, and the target
        """
        raise NotImplementedError

    @staticmethod
    def get_datasets(dataset_dictionary):
        """
        Method to extract the train, test, and valid sets

        :param dataset_dictionary: Python dictionary that contains the three sets

        :return: Python tuple containing the train, test, and valid sets
        """
        return dataset_dictionary["train"], dataset_dictionary["test"], dataset_dictionary["valid"]


class NNTrainer(Trainer):
    def __init__(self, model, metric, lr, batch_size, weight_decay, epochs, early_stopping_activated=False,
                 patience=10, device="cpu", trial=None, seed=None):
        """
        Creates a  Trainer that will train and evaluate a Neural Network model.

        :param batch_size: Int that represents the size of the batches to be used in the train data loader
        :param lr: the learning rate
        :param weight_decay: The L2 penalty
        :param epochs: Number of epochs to train the training dataset
        :param early_stopping_activated: Bool indicating if we want to early stop the training when the validation
        loss stops decreasing
        :param patience: Number of epochs without improvement allowed before early stopping
        ce: Int representing how long to wait after last time validation loss improved.
        :param device: The device where we want to run our training, this parameter can take two values : "cpu" or "gpu"
        :param model: Neural network model to be trained
        :param seed: The starting point in generating random numbers
        """
        super().__init__(model=model, metric=metric, device=device)

        assert isinstance(model, Module), 'model argument must inherit from torch.nn.Module'

        # we save the criterion of that model in the attribute criterion
        self.criterion = model.loss
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.early_stopping_activated = early_stopping_activated
        self.patience = patience
        self.seed = seed
        self.trial = trial

    def update_progress_func(self, trial, verbose):
        if trial is None and verbose:
            def update_progress(epoch, mean_epoch_loss):
                if epoch % 5 == 0 or (epoch + 1) == self.epochs:
                    print(f"Epoch {epoch + 1} - Loss : {round(mean_epoch_loss, 4)}")
        else:
            def update_progress(**kwargs):
                pass

        return update_progress

    def fit(self, train_set, val_set, verbose=True):
        """
        Method that will fit the model to the given data

        :param train_set: Petale Dataset containing the training set
        :param val_set: Petale Dataset containing the valid set
        :param verbose: Determines if we want to print progress or not


        :return: Two python lists containing the training losses and the validation losses
        """
        assert not (self.trial is not None and self.metric is None), "If trial is not None, a metric must be defined"
        assert self.model is not None, "Model must be set before training"

        if self.seed is not None:
            manual_seed(self.seed)

        # The maximum value of the batch size is the size of the train set
        if len(train_set) < self.batch_size:
            self.batch_size = len(train_set)

        # We create the train data loader
        if len(train_set) % self.batch_size == 1:
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # We create the optimizer with SWA
        base_optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = SWA(base_optimizer, swa_start=10, swa_freq=5)

        # We initialize two empty lists to store the training loss and the validation loss
        training_loss, valid_loss = [], []

        # We init the early stopping class
        early_stopping = EarlyStopping(patience=self.patience)

        # We init the update function
        update_progress = self.update_progress_func(self.trial, verbose)

        # We send model to device
        self.model.to(self.device)

        for epoch in range(self.epochs):

            ###################
            # train the model #
            ###################

            # We calculate training mean epoch loss on all batches
            mean_epoch_loss = self.train(train_loader, optimizer)
            update_progress(epoch=epoch, mean_epoch_loss=mean_epoch_loss)

            # We record training loss
            training_loss.append(mean_epoch_loss)

            ######################
            # validate the model #
            ######################

            # We calculate validation epoch loss and save it
            val_epoch_loss = self.evaluate(val_set, early_stopping)
            valid_loss.append(val_epoch_loss)

        return tensor(training_loss), tensor(valid_loss)

    def predict(self, x_cont, x_cat, **kwargs):

        """
        Returns log probabilities in the case of an NNClassifier
        Returns real-valued targets in the case of NNRegressor

        :param x_cont: tensor with continuous inputs
        :param x_cat: tensor with categorical ordinal encoding
        :return: (N, 1) or (N, C) tensor
        """

        # We return the predictions
        return self.model.predict(x_cont, x_cat, **kwargs)

    def train(self, train_loader, optimizer):
        """
        Trains the model for a single epoch

        :param train_loader: Training DataLoader
        :param optimizer: PyTorch optimizer
        :return: mean epoch loss
        """

        # Prep model for training
        self.model.train()
        epoch_loss = 0

        for item in train_loader:

            # We extract the continuous data x_cont, the categorical data x_cat
            # and the correct predictions y
            x_cont, x_cat, y = self.extract_batch(item)

            # We clear the gradients of all optimized variables
            optimizer.zero_grad()

            # We perform the forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(x_cont=x_cont, x_cat=x_cat)

            # We calculate the loss
            loss = self.criterion(output, y)
            epoch_loss += loss.item()

            # We perform the backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # We perform a single optimization step (parameter update)
            optimizer.step()

        return epoch_loss / len(train_loader)

    def evaluate(self, valid_set, early_stopper):

        """
        Calculates the loss on the validation set using a single batch
        There will be no memory problem since our datasets are really small

        :param valid_set: Validation PetaleDataset
        :param early_stopper: EarlyStopping object
        :return: Validation loss
        """
        # Prep model for validation
        self.model.eval()

        with torch.no_grad():

            # We extract the continuous data x_cont, the categorical data x_cat
            # and the correct predictions y for the single batch
            x_cont, x_cat, y = self.extract_data(valid_set)

            # We perform the forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(x_cont=x_cont, x_cat=x_cat)

            # We calculate the loss
            val_epoch_loss = self.criterion(output, y).item()

        # We calculate a score for the current model
        # score = self.metric(self.model.predict(x_cont=x_cont, x_cat=x_cat, log_prob=True), y)

        # We look for early stopping
        if self.early_stopping_activated:
            early_stopper(val_epoch_loss, self.model)
            if early_stopper.early_stop:
                self.model = early_stopper.get_best_model()

        return val_epoch_loss

    def extract_data(self, dataset):
        """
        Method to extract the continuous data, categorical data, and the target

        :param dataset: PetaleDataset or PetaleDataframe containing the data
        :return: Python tuple containing the continuous data, categorical data, and the target
        """
        x_cont, y = dataset.X_cont, dataset.y

        if dataset.X_cat is not None:
            x_cat = dataset.X_cat
        else:
            x_cat = None

        # We send all data to the device
        x_cont, x_cat, y = x_cont.to(self.device), x_cat.to(self.device), y.to(self.device)

        return x_cont, x_cat, y

    def update_trainer(self, **kwargs):
        """
        Updates the model and the weight decay
        """
        self.model = kwargs.get('model', self.model)
        self.weight_decay = kwargs.get('weight_decay', self.weight_decay)


class RFTrainer(Trainer):

    def __init__(self, model, metric):
        """
        Creates a  Trainer that will train and evaluate a Random Forest model.

        :param model: the model to be trained
        """
        super().__init__(model=model, metric=metric)

    def fit(self, train_set, **kwargs):
        """
        Method that will fit the model to the given data

        :param train_set: Pandas dataframe containing the training set
        """

        self.model.fit(train_set.X_cont, train_set.y)

    def predict(self, x_cont, x_cat=None, **kwargs):

        # We return the log probabilities
        return self.model.predict_log_proba(x_cont)

    def extract_data(self, dataset):
        """
        Method to extract the continuous data, categorical data, and the target

        :param dataset: PetaleDataset or PetaleDataframe containing the data
        :return: Python tuple containing the continuous data, categorical data, and the target
        """
        x_cont, y = dataset.X_cont, dataset.y

        if dataset.X_cat is not None:
            x_cat = dataset.X_cat
        else:
            x_cat = None

        return x_cont, x_cat, y

    def update_trainer(self, **kwargs):
        """
        Updates the model and the weight decay
        """
        self.model = kwargs.get('model', self.model)


def get_kfold_data(dataset, k, i):
    """
        Function that will be used to extract the fold needed

        :param dataset: Petale Dataset containing the data
        :param k: Number of folds
        :param i: the index of the fold that will  represent the validation set

        :return: returns two subset of the dataset, one for the training set and one for the validation set
    """
    # we check some conditions before going further
    assert k > 1
    assert i < k

    # We get the size of one fold
    fold_size = len(dataset) // k

    # We initialize a list that will contain the all the indexes of the the items that will be in the training set
    train_idx = []
    for j in range(k):

        # We get all the indexes of the items of the current fold
        idx = range(fold_size * j, fold_size * (j + 1))
        if i == j:
            # We save the indexes of the items of the validation set
            valid_idx = idx
        else:
            # We save the indexes of the items of the training set
            train_idx += idx
    # We return two subsets of the dataset, one representing the training set and one representing the validation set
    return Subset(dataset, train_idx), Subset(dataset, list(valid_idx))
