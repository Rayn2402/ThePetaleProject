"""
Authors : Mehdi Mitiche

File that contains the class that will be responsible of generating the model when we tune the hyper parameters

"""

class ModelGenerator():
    def __init__(self, modelClass, num_cont_col, cat_sizes = None, output_size = None ):
        """
        Class that will be responsible of geerating the model
        
        :param modelClass: class of the model we want to use 
        :param num_cont_col: the number of continuous columns we have
        :param cat_sizes: list of integer representing the size of each categorical column
        :param output_size: the number of nodes in the last layer of the neural network or the the number of classes
        """
        self.modelClass = modelClass
        self.num_cont_col = num_cont_col
        self.cat_sizes = cat_sizes
        self.output_size = output_size
    def __call__(self, layers, dropout):
        """
        The method to call to generate the model

        :param layers: a list to represent the number of hidden layers and the number of units in each layer
        :param dropout: a fraction representing the probability of dropout

        :return: a model
        """
        if(self.output_size is None):
            # the case when the model dosn't need the the parameter output size like the model NNRegressor
            return self.modelClass(num_cont_col = self.num_cont_col, cat_sizes=self.cat_sizes,layers=layers, dropout=dropout)
        else:
            # the case when the model need the the parameter output size
            return self.modelClass(num_cont_col = self.num_cont_col, cat_sizes=self.cat_sizes, output_size = self.output_size,layers=layers, dropout=dropout)