"""
Authors : Mehdi Mitiche

This file stores the two classes of Neural Networks models : 
NNRegressor which is a model to preform a regression and predict a real value
"""

from torch import  cat, argmax
from torch.nn import Module, ModuleList, Embedding, Linear, MSELoss, ReLU, BatchNorm1d, Dropout, Sequential, CrossEntropyLoss

class NNModel(Module):
    def __init__(self, num_cont_col , output_size, layers, dropout = 0.4, cat_sizes=None):
        """
        Creates a Neural Network model, entity embedding
        is performed on the categorical data if cat_sizes is not null

        :param num_cont_col: the number of continuous columns we have
        :param output_size: the number of nodes in the last layer of the neural network or the the number of classes
        :param layers: a list to represent the number of hidden layers and the number of units in each layer
        :param dropout: a fraction representing the probability of dropout
        :param cat_sizes: list of integer representing the size of each categorical column
        """
        super(NNModel, self).__init__()
        if cat_sizes is not None:

            # we generate the embedding sizes ( this part will be optimized )
            embedding_sizes = [(cat_size, min(50, (cat_size+1)//2)) for cat_size in cat_sizes]

            # we create the Embeddings layers
            self.embedding_layers = ModuleList([Embedding(num_embedding, embedding_dim) for
                                                   num_embedding, embedding_dim in embedding_sizes])

            # we get the number of our categorical data after the embedding ( we sum the embeddings dims)
            num_cat_col = sum((embedding_dim for num_embedding, embedding_dim in embedding_sizes))
            
            # the number of enteries to our linear layer
            input_size = num_cat_col + num_cont_col
        else:
            # the number of enteries to our linear layer
            input_size = num_cont_col

        # we intialize an empty list that will contin all the layers of our model
        all_layers = []

        #we create the diffrent layers of our model : Linear --> ReLU --> Batch Normalization --> Dropout
        for i in layers:
            # Linear Layer
            all_layers.append(Linear(input_size, i))
            # Activatiin function
            all_layers.append(ReLU(inplace=True))
            # batch normalization
            all_layers.append(BatchNorm1d(i))
            # dropour layer
            all_layers.append(Dropout(dropout))
            input_size = i 
        
        #we define the output layer
        if len(layers) == 0 :
            all_layers.append(Linear(input_size, output_size))
        else :
            all_layers.append(Linear(layers[-1], output_size))

        # we save all our layers in self.layers
        self.layers = Sequential(*all_layers)

        #we define the criterion for that model
        #self.criterion = MSELoss()
    
    def forward(self, x_cont, x_cat=None):
        embeddings = []
        if x_cat is not None:
            # we perform the entity embedding
            for i, e in enumerate(self.embedding_layers):
                embeddings.append(e(x_cat[:, i].long()))
            # we concatenate all the embeddings
            x = cat(embeddings, 1)

            # we concatenate categorical and continuous data after performing the  entity embedding
            x = cat([x, x_cont], 1)
        else:
            x = x_cont
        return self.layers(x)
    """def loss(self, x_cont, x_cat, target):
        return ((self(x_cont.float(),x_cat).squeeze() - target)**2).mean().item()"""


class NNRegressor(NNModel):
    def __init__(self, num_cont_col, layers , dropout = 0.4, cat_sizes=None):
        super().__init__(num_cont_col= num_cont_col, output_size=1,layers= layers, dropout= dropout,cat_sizes= cat_sizes)

        #we define the criterion for that model
        self.criterion = MSELoss()
    
    def loss(self, x_cont, x_cat, target):
        return ((self(x_cont.float(),x_cat).squeeze() - target)**2).mean().item()
    


class NNClassifier(NNModel):
    def __init__(self, num_cont_col, output_size, layers, dropout = 0.4 ,cat_sizes=None):
        super().__init__(num_cont_col= num_cont_col, output_size=output_size,layers= layers, dropout= dropout,cat_sizes= cat_sizes)

        #we define the criterion for that model
        self.criterion = CrossEntropyLoss()
    
    def loss(self, x_cont, x_cat, target):  
        true_answers = 0
        predictions =(argmax(self(x_cont.float(),x_cat).float(), dim=1))
        for index, row in enumerate(predictions):
            if(row.item() == target[index].item()):
                true_answers += 1
        return true_answers / predictions.shape[0]
