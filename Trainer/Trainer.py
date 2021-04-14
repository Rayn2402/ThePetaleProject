"""
Authors : Mitiche

Files that contains class related to the Trainer of the models

"""
from Trainer.EarlyStopping import EarlyStopping
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch import optim, manual_seed, cuda, tensor
from torch import device as device_
from optuna import TrialPruned
from torchcontrib.optim import SWA


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
        self.device = device

        # We save the metric
        self.metric = metric

    def cross_valid(self, datasets, k=5, seed=None):
        """
            Method that will perform a cross validation on the model

            :param datasets: Petale Datasets representing all the train, test, and valid sets to be used in the cross
             validation
            :param k: Number of folds
            :param seed: the starting point in generating random numbers

            :return: The score after performing the cross validation
        """

        # Seed is left to None if fit is called by NNTuner
        if seed is not None:
            manual_seed(seed)

        # We initialize an empty list to store the scores
        score = []
        for i in range(k):
            # We the get the train, test, valid sets of the step we are currently in
            train_set, test_set, valid_set = self.get_datasets(datasets[i])

            # we train our model with the train and valid sets
            self.fit(train_set=train_set, val_set=valid_set)

            # We extract x_cont, x_cat and target from the test set
            x_cont, x_cat, target = self.extract_data(test_set)

            # We calculate the score with the help of the metric function
            intermediate_score = self.metric(self.predict(x_cont=x_cont, x_cat=x_cat), target)

            # We save the score
            score.append(intermediate_score)

        # We return the final score of the cross validation
        return sum(score) / len(score)

    @staticmethod
    def extract_batch(batch_list, device):
        """
        Extracts the continuous data (X_cont), the categorical data (X_cat) and the ground truth (y)

        :param batch_list: list containing a batch from dataloader
        :param device: "cpu" or "gpu"
        :return: 3 tensors
        """

        if len(batch_list) > 2:
            x_cont, x_cat, y = batch_list
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
        else:
            x_cont, y = batch_list
            x_cont, y = x_cont.to(device), y.to(device)
            x_cat = None

        return x_cont, x_cat, y

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
        self.criterion = model.criterion_function
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.early_stopping_activated = early_stopping_activated
        self.patience = patience
        self.seed = seed
        self.trial = trial
        self.best_model = None

    def update_progress_func(self, trial, verbose):
        if trial is None and verbose:
            def update_progress(epoch, mean_epoch_loss):
                if epoch % 10 == 0 or (epoch + 1) == self.epochs:
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
        assert not (self.trial is not None and self.metric is None)

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

        # We create the validation data loader
        val_loader = DataLoader(val_set, batch_size=len(val_set))

        # Ee create the optimizer
        base_optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # We implement ASWA
        optimizer = SWA(base_optimizer, swa_start=10, swa_freq=5)

        # We initialize two empty lists to store the training loss and the validation loss
        training_loss, valid_loss = [], []

        # We init the early stopping class
        early_stopping = EarlyStopping(patience=self.patience)

        # We init the update function
        update_progress = self.update_progress_func(self.trial, verbose)

        # We declare the variable which will hold the device we’re training on
        device = device_("cuda" if cuda.is_available() and self.device == "gpu" else "cpu")

        for epoch in range(self.epochs):

            ###################
            # train the model #
            ###################

            # Prep model for training
            self.model.train()
            epoch_loss = 0

            for item in train_loader:
                # We extract the continuous data x_cont, the categorical data x_cat
                # and the correct predictions y
                x_cont, x_cat, y = self.extract_batch(item, device)

                # We clear the gradients of all optimized variables
                optimizer.zero_grad()

                # We perform the forward pass: compute predicted outputs by passing inputs to the model
                preds = self.model(x_cont=x_cont, x_cat=x_cat)

                # We calculate the loss
                loss = self.criterion(preds, y)
                epoch_loss += loss.item()

                # We perfrom the backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # We perform a single optimization step (parameter update)
                optimizer.step()

            mean_epoch_loss = epoch_loss / len(train_loader)
            update_progress(epoch=epoch, mean_epoch_loss=mean_epoch_loss)

            # We record training loss
            training_loss.append(mean_epoch_loss)

            ######################
            # validate the model #
            ######################

            # Prep model for validation
            self.model.eval()

            # We extract the continuous data x_cont, the categorical data x_cat
            # and the correct predictions y for the single batch
            x_cont, x_cat, y = self.extract_batch(next(iter(val_loader)), device)

            # We perform the forward pass: compute predicted outputs by passing inputs to the model
            preds = self.model(x_cont=x_cont, x_cat=x_cat)

            # We calculate the loss
            val_epoch_loss = self.criterion(preds, y).item()

            # We record the validation loss
            valid_loss.append(val_epoch_loss)

            # We look for early stopping
            if self.early_stopping_activated:
                self.best_model = early_stopping(val_epoch_loss, self.model)

            if early_stopping.early_stop:
                self.model = self.best_model

            if self.metric is not None:
                intermediate_score = self.metric(self.model(x_cont=x_cont, x_cat=x_cat), y) if \
                    not self.early_stopping_activated else \
                    self.metric(self.best_model(x_cont=x_cont, x_cat=x_cat), y)

            # Pruning logic
            if self.trial is not None:
                # We report the score to optuna
                self.trial.report(intermediate_score, step=epoch)
                # We prune the trial if it should be pruned
                if self.trial.should_prune():
                    raise TrialPruned()

        return tensor(training_loss), tensor(valid_loss)

    def predict(self, x_cont, x_cat):

        # We set the model on evaluation mode
        self.model.eval()

        # We return the predictions
        return self.model(x_cont, x_cat).float()


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

    def predict(self, x_cont, x_cat=None):
        # We return the predictions
        return self.model.predict(x_cont)


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
    fold_size = dataset.__len__() // k

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
