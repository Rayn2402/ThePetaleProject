"""
Authors : Mitiche

Files that contains class related to the Training of the models

"""
from .EarlyStopping import EarlyStopping
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch import optim, manual_seed, cuda, tensor
from torch import device as device_
from tqdm import tqdm
from optuna import TrialPruned


class Trainer:

    def __init__(self, model):
        """
        Creates a Trainer that will train and evaluate a given model.

        :param model: the model to be trained
        """
        assert isinstance(model, Module), 'model argument must inherit from torch.nn.Module'

        # We save the model in the attribute model
        self.model = model

        # We save the criterion of that model in the attribute criterion
        self.criterion = model.criterion_function

    def fit(self, train_set, val_set, batch_size, lr, weight_decay, epochs, early_stopping_activated=True,
            patience=10, device="cpu", trial=None, metric=None, seed=None):
        """
        Method that will fit the model to the given data

        :param train_set: PetaleDataset containing the training set
        :param val_set: PetaleDataset containing the valid set
        :param batch_size: Int representing the size of the batches to be used in the train data loader
        :param lr: Learning rate
        :param weight_decay: L2 penalty
        :param epochs: Number of times that the learning algorithm will work through the entire training dataset
        :param early_stopping_activated: Boolean indicating if we want to early stop
                                         the training when the validation loss stops decreasing
        :param patience: Int representing how long to wait after last time validation loss improved.
        :param device: Device where we want to run our training, this parameter can take two values : "cpu" or "gpu"
        :param trial: Optuna Trial to report intermediate value
        :param metric: A function that takes the output of the model and the target and returns the metric we want to
                       measure
        :param seed: The starting point in generating random numbers

        :return: Two lists containing the training losses and the validation losses
        """

        assert (trial is None and metric is None) or (trial is not None and metric is not None)

        # Seed is left to None if fit his called by NNEvaluator
        if seed is not None:
            manual_seed(seed)

        # The maximum value of the batch size is the size of the trainset
        if len(train_set) < batch_size:
            batch_size = len(train_set)

        # We create the the train data loader
        if len(train_set) % batch_size == 1:
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        # We create the the validation data loader
        val_loader = DataLoader(val_set, batch_size=len(val_set))

        # We create the optimizer
        optimizer = optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # We initialize two empty lists to store the training loss and the validation loss
        training_loss, valid_loss = [], []

        # We init the early stopping object
        early_stopping = EarlyStopping(patience=patience)

        # we declare the variable which will hold the device we’re training on 
        device = device_("cuda" if cuda.is_available() and device == "gpu" else "cpu")

        # We declare a tqdm loading bar
        bar = tqdm(range(epochs), position=0, leave=True)
        bar.set_description(f'Epoch 0')
        bar.set_postfix_str(s=f"Loss : NaN")

        for epoch in range(epochs):

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

                # Clear the gradients of all optimized variables
                optimizer.zero_grad()

                # Forward pass: compute predicted outputs by passing inputs to the model
                preds = self.model(x_cont=x_cont, x_cat=x_cat)

                # Calculate the loss
                loss = self.criterion(preds, y)
                epoch_loss += loss.item()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimizer.step()

            mean_epoch_loss = epoch_loss / len(train_loader)
            bar.set_description(f'Epoch {epoch}')
            bar.set_postfix_str(s=f"Loss : {round(mean_epoch_loss, 4)}")
            bar.update()

            # Record training loss
            training_loss.append(mean_epoch_loss)

            ######################
            # validate the model #
            ######################

            # Prep model for validation
            self.model.eval()

            # We extract the continuous data x_cont, the categorical data x_cat
            # and the correct predictions y for the single batch
            x_cont, x_cat, y = self.extract_batch(next(iter(val_loader)), device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            preds = self.model(x_cont=x_cont, x_cat=x_cat)

            # Calculate the loss
            val_epoch_loss = self.criterion(preds, y).item()

            # Record validation loss
            valid_loss.append(val_epoch_loss)

            if metric is not None:
                intermediate_score = metric(preds, y)

            if trial is not None:

                # we report the score to optuna
                trial.report(intermediate_score, step=epoch)

                # we prune the trial if it should be pruned
                if trial.should_prune():
                    raise TrialPruned()

            # We look for early stopping
            if early_stopping_activated:
                early_stopping(val_epoch_loss, self.model)

            if early_stopping.early_stop:
                break

        return tensor(training_loss), tensor(valid_loss)

    def cross_valid(self, datasets, batch_size, lr, weight_decay, epochs, metric, k=5, early_stopping_activated=False,
                    patience=5, trial=None, seed=None):
        """
        Method that will perform a k-fold cross validation on the model

        :param datasets: Petale Datasets representing all the train and test sets to be used in the cross validation
        :param batch_size: int that represent the size of the batches to be used in the train data loader
        :param lr: the learning rate
        :param weight_decay: the L2 penalty
        :param k: number of folds
        :param metric: a function that takes the output of the model and the target and returns the metric we want to
        measure
        :param epochs: number times that the learning algorithm will work through the entire training dataset
        :param early_stopping_activated: boolean indicating if we want to early stop the training when the validation
        loss stops decreasing
        :param patience: int representing how long to wait after last time validation loss improved.
        :param trial: Optuna Trial to report intermediate value
        :param seed: the starting point in generating random numbers


        :return: returns the score after performing the k-fold cross validation
        """
        # Seed is left to None if fit his called by NNTuner
        if seed is not None:
            manual_seed(seed)

        # we initialize an empty list to store the scores
        score = []
        for i in range(k):

            # we the get the train and the validation datasets of the step we are currently in
            train_set, valid_set, test_set = datasets[i]["train"], datasets[i]["valid"], datasets[i]["test"]

            # we train our model with this train and validation dataset
            self.fit(train_set=train_set, val_set=valid_set, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                     epochs=epochs, early_stopping_activated=early_stopping_activated, patience=patience,
                     trial=trial, metric=metric)

            # we extract x_cont, x_cat and target from the subset valid_fold
            x_cont = test_set.X_cont
            target = test_set.y
            if test_set.X_cat is not None:
                x_cat = test_set.X_cat
            else:
                x_cat = None

            # we calculate the score with the help of the metric function
            intermediate_score = metric(self.model(x_cont, x_cat).float(), target)

            # we save the score
            score.append(intermediate_score)

        # we return the final score of the cross validation
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
