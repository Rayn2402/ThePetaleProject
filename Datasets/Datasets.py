"""
Authors : Nicolas Raymond

Files that contains class related to Datasets

"""

from torch.utils.data import Dataset
from torch import from_numpy, cat
from .Preprocessing import *
from .Transforms import ContinuousTransform as ConT


class PetaleDataset(Dataset):

    def __init__(self, df, cont_cols, target, cat_cols=None):
        """
        Creates a petale dataset where categoricals columns are separated from the continuous by default.

        :param df: pandas dataframe
        :param cont_cols: list with names of continuous columns
        :param target: string with target column name
        :param cat_cols: list with names of categorical columns
        """

        self.X_cont = preprocess_continuous(df[cont_cols])
        self.X_cont = from_numpy(self.X_cont.values)

        if cat_cols is not None:
            self.X_cat, self.encoding_sizes = preprocess_categoricals(df[cat_cols])
            self.X_cat = from_numpy(self.X_cat.values)

        self.y = from_numpy(ConT.to_float(df[target]).values).flatten()
        self.getter = self.define_getter(cat_cols)

    def __len__(self):
        return self.X_cont.shape[0]

    def __getitem__(self, idx):
        return self.getter(idx)

    def define_getter(self, cat_cols):
        """
        Builds to right __getitem__ function according to cat_cols value

        :param cat_cols: list of categorical columns names
        :return: function
        """
        if cat_cols is not None:
            def f(idx):
                return self.X_cont[idx, :], self.X_cat[idx, :], self.y[idx]

        else:
            def f(idx):
                return self.X_cont[idx, :], self.y[idx]

        return f

    def concat_dataset(self):
        """
        Concatenates categorical and continuous data
        WARNING! : !Categorical and continuous must remain separate if we want to have an embedding layer!
        """
        self.X_cont = cat((self.X_cont, self.X_cat), 1)
        self.X_cat, self.encoding_sizes = None, None
        self.getter = self.define_getter(None)




