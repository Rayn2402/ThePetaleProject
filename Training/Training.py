from .EarlyStopping import EarlyStopping
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm

# optimizers that can be used (Other optiizers could be added here)
optimizers = ["Adam", "RMSprop", "SGD"]

class Trainer():
    def __init__(self, model):
        if not isinstance(model, nn.Module):
            raise ValueError('model argument must inherit from torch.nn.Module')
        self.model = model
    
    def fit(self, dataset,train_split , batch_size, optimizer_name, lr, epochs, early_stopping_activated = True, patience = 5):
        #we split the data
        train_split = int(train_split * dataset.__len__())
        train_set, val_set = random_split(dataset, [train_split, dataset.__len__() - train_split])
        #we create the the train data loader
        train_loader = DataLoader(train_set,batch_size,shuffle=True)
        val_loader = DataLoader(val_set,batch_size= val_set.__len__())
        #we create the the validation data loader
        train_loader = DataLoader(dataset,batch_size,shuffle=True)
        #we create the 
        if optimizer_name not in optimizers:
            raise Exception("optimizer not found !")
        optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters(), lr=lr)
        #we create the criterion
        criterion = nn.MSELoss()
        training_loss = []
        valid_loss = []

        if early_stopping_activated:
            #we init the early stopping class
            early_stopping = EarlyStopping(patience=patience)

        for epoch in tqdm(range(epochs)):
            ###################
            # train the model #
            ###################
            # prep model for training
            self.model.train()
            epoch_loss = 0
            for x, y in train_loader:
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                preds = self.model(x.float()).flatten()
                # calculate the loss
                loss = criterion(preds, y.float())
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
            for x, y in val_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                preds = self.model(x.float()).flatten()
                y = y.type(torch.float32)
                # calculate the loss
                loss = criterion(preds, y)
                val_epoch_loss += loss.item()
            # record training loss
            valid_loss.append(val_epoch_loss / len(val_loader))

            if early_stopping_activated :
                #early stopping
                early_stopping(val_epoch_loss / len(val_loader),self.model)
                if(early_stopping.early_stop):
                    print("Early stopping ...")
                    break
        return training_loss,valid_loss
    
    def predict(self, x):
        return self.model(x.float())
    def loss(self,x ,target):
        return ((self.predict(x).unsqueeze(dim=0) - target)**2).mean().item()
        



