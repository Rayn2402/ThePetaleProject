"""
Authors : Mitiche

Files that contains class related to the Training of the models

"""

from .EarlyStopping import EarlyStopping
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch import optim
from tqdm import tqdm

# optimizers that can be used (Other optiizers could be added here)
optimizers = ["Adam", "RMSprop", "SGD"]

class Trainer():
    def __init__(self, model):
        """
        Creates a Trainer that will train and evaluate a given model.

        :param model: the model to be trained
        """
        if not isinstance(model, Module):
            raise ValueError('model argument must inherit from torch.nn.Module')
        #we save the model in the attribute model
        self.model = model
        #we save the criterion of that model in the attribute criterion
        self.criterion = model.criterion
    
    def fit(self, train_set, val_set, batch_size, optimizer_name, lr, epochs, early_stopping_activated = True, patience = 5):
        """
        Method that will fit the model to the given data

        :param train_set: Petale Dataset containing the training set
        :param test_set: Petale Dataset containing the test data
        :param batch_size: int that represent the size of the batchs to be used in the train data loader
        :param optimizer_name: string to define the optimizer to be used in the training
        :param lr: the learning rate
        :param epochs: number times that the learning algorithm will work through the entire training dataset
        :param early_stopping_activated: boolean indicating if we want to early stop the training when the validation loss stops decreasing
        :param patience: int representing how long to wait after last time validation loss improved.

        :return: two lists containing the training losses and the validation losses
        """
        #we create the the train data loader
        train_loader = DataLoader(train_set,1,shuffle=True)
        #we create the the validation data loader
        val_loader = DataLoader(val_set,batch_size=val_set.__len__())
        #we create the optimizer
        if optimizer_name not in optimizers:
            raise Exception("optimizer not found !")
        optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr)
        #we initialize two empty lists to store the training loss and the validation loss
        training_loss = []
        valid_loss = []

        #we init the early stopping class
        early_stopping = EarlyStopping(patience=patience)
        
        for epoch in tqdm(range(epochs)):
            ###################
            # train the model #
            ###################
            # prep model for training
            self.model.train()
            epoch_loss = 0
            for item in train_loader:
                # y will contains the correct prediction
                y = item[-1]
                # x will contain both continuous data and categorical data if there is
                x = item[:-1]
                # we traansform the x data to float : TO BE UPDATED
                x = map(lambda x: x.float(), x)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                preds = self.model(*x).flatten()
                # calculate the loss
                loss = self.criterion(preds, y.float())
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
                # we traansform the x data to float : TO BE UPDATED
                x = map(lambda x: x.float(), x)               
                # forward pass: compute predicted outputs by passing inputs to the model
                preds = self.model(*x).flatten()
                # calculate the loss
                loss = self.criterion(preds, y)
                val_epoch_loss += loss.item()
            # record training loss
            valid_loss.append(val_epoch_loss / len(val_loader))

            if early_stopping_activated :
                early_stopping(val_epoch_loss / len(val_loader),self.model)
            if(early_stopping.early_stop):
                break
        return training_loss, valid_loss
    
    def cross_valid(self, dataset, batch_size,optimizer_name, lr, epochs,  k=5, early_stopping_activated = True, patience=5):
        """
        Method that will perfrom a k-fold cross validation on the model

        :param dataset: Petale Dataset containing the data
        :param batch_size: int that represent the size of the batchs to be used in the train data loader
        :param optimizer_name: string to define the optimizer to be used in the training
        :param lr: the learning rate
        :param k: number of folds
        :param epochs: number times that the learning algorithm will work through the entire training dataset
        :param early_stopping_activated: boolean indicating if we want to early stop the training when the validation loss stops decreasing
        :param patience: int representing how long to wait after last time validation loss improved.

        :return: returns the score after performing the k-fold cross validation
        """
        # we initialize an empty list to store the scores
        score = []
        for i in range(k):
            # we the get the train and the validation datasets of the step we are currently in
            k_fold_train, k_fold_valid = get_kfold_data(dataset, k, i)
            # we train our model with this train and validation dataset
            train_loss, valid_loss = self.fit(k_fold_train, k_fold_valid, batch_size, optimizer_name, lr, epochs, early_stopping_activated=early_stopping_activated, patience=patience)
            # we save the score of this training
            score.append(valid_loss[-1])
        # we return the final score of the cross validation
        return sum(score)/len(score)

def get_kfold_data(dataset, k, i):
    """
        Function that will be used to extract the fold needed

        :param dataset: Petale Dataset containing the data
        :param k: number of folds
        :param i: the index of the fold that will  represent the validation set

        :return: returns two subset of the dataset, one for the training set and one for the validation set
    """
    # we check some condtions before going further
    assert k>1  
    assert i<k
    # we get the size of one fold
    fold_size = dataset.__len__()//k
    #we initialize a list that will contain the all the indexes of the the items that will be in the training set
    train_idx = []
    for j in range(k):
        # we get all the indexes of the items of the current fold
        idx = range(fold_size*j, fold_size*(j+1))
        if i==j:
            # we save the indexes of the items of the validation set
            valid_idx = idx
        else:
            # we save the indexes of the items of the training set
            train_idx += idx
    # we return two subsets of the dataset, one representing the training set and one representing the validation set
    return Subset(dataset, train_idx), Subset(dataset,list(valid_idx))
            
