"""
Authors : Mitiche

Files that contains class related to the Training of the models

"""

from .EarlyStopping import EarlyStopping
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch import optim, manual_seed, cuda
from torch import device as device_
from tqdm import tqdm
from optuna import TrialPruned


class Trainer:
    def __init__(self, model, device="cpu"):
        """
        Creates a Trainer that will train and evaluate a given model.

        :param model: the model to be trained
        :param device: the device where we want to run our training, this parameter can take two values : "cpu" or "gpu"

        """
        # we save the model in the attribute model
        self.model = model

    def cross_valid(self, datasets, metric, k=5, trial=None):
        """
            Method that will perform a k-fold cross validation on the model

            :param datasets: Petale Datasets representing all the train and test sets to be used in the cross validation
            :param k: number of folds
            :param metric: a function that takes the output of the model and the target and returns the metric we want to
            measure
            :param trial: Optuna Trial to report intermediate value

            :return: returns the score after performing the k-fold cross validation
        """

        # we initialize an empty list to store the scores
        score = []
        for i in range(k):
            # we the get the train and the validation datasets of the step we are currently in
            if len(datasets[i]) > 2:
                train_set, valid_set, test_set = datasets[i]["train"], datasets[i]["valid"], datasets[i]["test"]
            else:
                train_set, test_set = datasets[i]["train"], datasets[i]["test"]
                valid_set = None

            # we train our model with this train and validation dataset
            self.fit(train_set=train_set, val_set=valid_set)

            # we extract x_cont, x_cat and target from the subset valid_fold
            x_cont = test_set.X_cont
            target = test_set.y
            if test_set.X_cat is not None:
                x_cat = test_set.X_cat
            else:
                x_cat = None

            # we calculate the score with the help of the metric function
            intermediate_score = metric(self.predict(x_cont=x_cont, x_cat=x_cat), target)

            if trial is not None:
                # we report the score to optuna
                trial.report(intermediate_score, step=i)
                # we prune the trial if it should be pruned
                if trial.should_prune():
                    raise TrialPruned()
            # we save the score
            score.append(intermediate_score)

        # we return the final score of the cross validation
        return sum(score) / len(score)


class NNTrainer(Trainer):
    def __init__(self, model, lr, batch_size, weight_decay, epochs, early_stopping_activated=True,
                 patience=5, seed=None, device="cpu"):
        """
        Creates a  Trainer that will train and evaluate a Neural Network model.

        :param batch_size: int that represent the size of the batches to be used in the train data loader
        :param lr: the learning rate
        :param weight_decay: the L2 penalty
        :param epochs: number times that the learning algorithm will work through the entire training dataset
        :param early_stopping_activated: boolean indicating if we want to early stop the training when the validation
        loss stops decreasing
        :param patience: int representing how long to wait after last time validation loss improved.
        :param seed: the starting point in generating random numbers
        :param device: the device where we want to run our training, this parameter can take two values : "cpu" or "gpu"

        :param model: the model to be trained
        """

        super().__init__(model, device=device)
        if not isinstance(self.model, Module):
            raise ValueError('model argument must inherit from torch.nn.Module')

        # we save the criterion of that model in the attribute criterion
        self.criterion = model.criterion_function
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.early_stopping_activated = early_stopping_activated
        self.patience = patience
        self.seed = seed

    def fit(self, train_set, val_set):
        """
        Method that will fit the model to the given data

        :param train_set: Petale Dataset containing the training set
        :param val_set: Petale Dataset containing the valid set


        :return: two lists containing the training losses and the validation losses
        """

        if self.seed is not None:
            manual_seed(self.seed)

        # the maximum value of the batch size is the size of the trainset
        if train_set.__len__() < self.batch_size:
            batch_size = train_set.__len__()

        # we create the the train data loader
        if len(train_set) % self.batch_size == 1:
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # we create the the validation data loader
        val_loader = DataLoader(val_set, batch_size=val_set.__len__())

        # we create the optimizer
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # we initialize two empty lists to store the training loss and the validation loss
        training_loss = []
        valid_loss = []

        # we init the early stopping class
        early_stopping = EarlyStopping(patience=self.patience)
        # we declare the variable which will hold the device we’re training on 
        device = device_("cuda" if cuda.is_available() and self.device == "gpu" else "cpu")

        for epoch in tqdm(range(self.epochs)):
            ###################
            # train the model #
            ###################

            # prep model for training
            self.model.train()

            epoch_loss = 0
            for item in train_loader:
                # we extract the continuous data x_cont, the categorical data x_cat and the  correct predictions y
                if len(item) > 2:
                    x_cont, x_cat, y = item
                    # we transfer the tensors to our device
                    x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
                else:
                    x_cont, y = item
                    # we transfer the tensors to our device
                    x_cont, y = x_cont.to(device), y.to(device)
                    x_cat = None

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                preds = self.model(x_cont=x_cont, x_cat=x_cat)

                # calculate the loss
                loss = self.criterion(preds, y)
                epoch_loss += loss.item()

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()
            # record training loss
            training_loss.append(epoch_loss / len(train_loader))

            ######################
            # validate the model #
            ######################

            # prep model for validation
            self.model.eval()
            val_epoch_loss = 0

            for item in val_loader:
                # y will contains the correct prediction
                y = item[-1]

                # x will contain both continuous data and categorical data if there is
                x = item[:-1]

                # we transform the x data to float : TO BE UPDATED
                x = map(lambda x: x.float(), x)

                # forward pass: compute predicted outputs by passing inputs to the model
                preds = self.model(*x)

                # calculate the loss
                loss = self.criterion(preds, y)
                val_epoch_loss += loss.item()
            # record training loss
            valid_loss.append(val_epoch_loss / len(val_loader))

            if self.early_stopping_activated:
                early_stopping(val_epoch_loss / len(val_loader), self.model)
            if early_stopping.early_stop:
                break
        return training_loss, valid_loss

    def predict(self, x_cont, x_cat):
        return self.model(x_cont, x_cat).float()


class RFTrainer(Trainer):
    def __init__(self, model):
        """
        Creates a  Trainer that will train and evaluate a Random Forest model.

        :param model: the model to be trained
        """
        super().__init__(model)

    def fit(self, train_set, valid_set=None):
        """
        Method that will fit the model to the given data

        :param train_set: Pandas dataframe containing the training set
        """
        self.model.fit(train_set.X_cat, train_set.Y)

    def predict(self, x_cont, x_cat = None):
        return self.model.predict(x_cont)



def get_kfold_data(dataset, k, i):
    """
        Function that will be used to extract the fold needed

        :param dataset: Petale Dataset containing the data
        :param k: number of folds
        :param i: the index of the fold that will  represent the validation set

        :return: returns two subset of the dataset, one for the training set and one for the validation set
    """
    # we check some conditions before going further
    assert k > 1
    assert i < k

    # we get the size of one fold
    fold_size = dataset.__len__() // k

    # we initialize a list that will contain the all the indexes of the the items that will be in the training set
    train_idx = []
    for j in range(k):

        # we get all the indexes of the items of the current fold
        idx = range(fold_size * j, fold_size * (j + 1))
        if i == j:
            # we save the indexes of the items of the validation set
            valid_idx = idx
        else:
            # we save the indexes of the items of the training set
            train_idx += idx
    # we return two subsets of the dataset, one representing the training set and one representing the validation set
    return Subset(dataset, train_idx), Subset(dataset, list(valid_idx))
