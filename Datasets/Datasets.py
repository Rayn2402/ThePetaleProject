"""
Authors : Nicolas Raymond

Files that contains class related to Datasets

"""

from torch.utils.data import Dataset
from torch import from_numpy, cat, ones
from .Preprocessing import *
from .Transforms import ContinuousTransform as ConT
from .Sampling import split_train_test
from SQL.NewTablesScripts.constants import *


class PetaleDataset(Dataset):

    def __init__(self, df, cont_cols, target, id, cat_cols=None,
                 split=True, add_biases=False, mean=None, std=None):
        """
        Creates a petale dataset where categoricals columns are separated from the continuous by default.

        :param df: pandas dataframe
        :param cont_cols: list with names of continuous columns
        :param target: string with target column name
        :param id: string with ids column name
        :param cat_cols: list with names of categorical columns
        :param split: boolean indicating if categorical features must remain separated from continuous features
        :param add_biases: boolean indicating if a column of ones should be added at the beginning of X_cont
        :param mean: means to use for data normalization (pandas series)
        :param std : stds to use for data normalization (pandas series)
        """
        if id not in df.columns:
            raise Exception('IDs missing from the dataframe')

        # We save the survivors ID
        self.IDs = df[id]

        # We save and preprocess continuous features
        self.X_cont = preprocess_continuous(df[cont_cols], mean, std)
        self.X_cont = from_numpy(self.X_cont.values)

        # We save the number of elements in the datasets
        self.N = self.IDs.shape[0]

        # We add biases to continuous features if required
        if add_biases:
            self.X_cont = cat((ones(self.N, 1), self.X_cont), 1)

        # We preprocess and save categorical features if there are some
        if cat_cols is not None:
            if split:
                # We keep ordinal encodings of categorical features separated from continuous features
                self.X_cat = preprocess_categoricals(df[cat_cols])
                self.X_cat = from_numpy(self.X_cat.values)
            else:
                # We concatenate one-hot encodings of categorical features with continuous features
                self.X_cat = preprocess_categoricals(df[cat_cols], encoding='one-hot')
                self.X_cat = from_numpy(self.X_cat.values)
                self.__concat_dataset()
        else:
            self.X_cat = None

        # We save the targets
        self.y = from_numpy(ConT.to_float(df[target]).values).flatten()

        # We define the getter function according to the presence of absence of categorical features
        self.getter = self.define_getter(cat_cols, split)

    def __len__(self):
        return self.X_cont.shape[0]

    def __getitem__(self, idx):
        return self.getter(idx)

    def define_getter(self, cat_cols, split):
        """
        Builds to right __getitem__ function according to cat_cols value

        :param cat_cols: list of categorical columns names
        :param split: boolean indicating if categorical features must remain separated from continuous features
        :return: function
        """
        if (cat_cols is not None) and split:
            def f(idx):
                return self.X_cont[idx, :], self.X_cat[idx, :], self.y[idx]

        else:
            def f(idx):
                return self.X_cont[idx, :], self.y[idx]

        return f

    def __concat_dataset(self):
        """
        Concatenates categorical and continuous data
        WARNING! : !Categorical and continuous must remain separate if we want to have an embedding layer!
        """
        self.X_cont = cat((self.X_cont, self.X_cat), 1)
        self.X_cat = None


def load_warmup_dataset(dm, test_size=0.15, split_cat=True, add_biases=False):
    """
    Loads 'Learning_0' table and create a PetaleDataset object

    :param dm: PetaleDataManager
    :param test_size: number of elements in the test set (if 0 < test_size < 1 we consider the parameter as a %)
    :param split_cat: Boolean indicating if we want to split categorical variables from the continuous ones
    :param add_biases: boolean indicating if a column of ones should be added at the beginning of X_cont
    :return: 2 PetaleDataset
    """
    # We save some important constants
    continuous_columns = [WEIGHT, TDM6_HR_END, TDM6_DIST, DT, AGE, MVLPA]

    # We load the data
    df = dm.get_table(LEARNING_0)

    # We make sure that continuous variables are considered as continuous
    df[continuous_columns] = ConT.to_float(df[continuous_columns])

    # We split the training and test data
    train, test = split_train_test(df, VO2R_MAX, test_size)

    # We save the mean and the standard deviations of the continuous columns in train
    mean, std = train[continuous_columns].mean(), train[continuous_columns].std()

    # We create the test and train datasets
    train_ds = PetaleDataset(train, continuous_columns, VO2R_MAX, PARTICIPANT,
                             split=split_cat, add_biases=add_biases)

    test_ds = PetaleDataset(test, continuous_columns, VO2R_MAX, PARTICIPANT,
                            split=split_cat, mean=mean, std=std, add_biases=add_biases)

    return train_ds, test_ds
