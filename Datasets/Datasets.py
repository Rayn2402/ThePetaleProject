"""
Authors : Nicolas Raymond

Files that contains class related to Datasets

"""

from torch.utils.data import Dataset
from torch import from_numpy, cat, ones
from .Preprocessing import *
from .Transforms import ContinuousTransform as ConT


class PetaleDataset(Dataset):

    def __init__(self, df, cont_cols, target, id, cat_cols=None, split=True, add_biases=False):
        """
        Creates a petale dataset where categoricals columns are separated from the continuous by default.

        :param df: pandas dataframe
        :param cont_cols: list with names of continuous columns
        :param target: string with target column name
        :param id: string with ids column name
        :param cat_cols: list with names of categorical columns
        :param split: boolean indicating if categorical features must remain separated from continuous features
        :param add_biases: boolean indicating if a column of ones should be added at the beginning of X_cont
        """
        if id not in df.columns:
            raise Exception('IDs missing from the dataframe')

        # We save the survivors ID
        self.IDs = df[id]

        # We save and preprocess continuous features
        self.X_cont = preprocess_continuous(df[cont_cols])
        self.X_cont = from_numpy(self.X_cont.values)

        # We save the number of elements in the datasets
        self.N = self.IDs.shape[0]

        # We add biases to continuous features if required
        if add_biases:
            self.X_cont = cat((ones(self.N, 1), self.X_cont), 1)

        # We preprocess and save categorical features if there are some
        if cat_cols is not None:
            self.X_cat, self.encoding_sizes = preprocess_categoricals(df[cat_cols])
            self.X_cat = from_numpy(self.X_cat.values)

        # We save the targets
        self.y = from_numpy(ConT.to_float(df[target]).values).flatten()

        # We define the getter function according to the presence of absence of categorical features
        self.getter = self.define_getter(cat_cols)

        # We consider all the dataset as continuous input if split == False
        if (not split) and cat_cols is not None:
            self.__concat_dataset()

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

    def __concat_dataset(self):
        """
        Concatenates categorical and continuous data
        WARNING! : !Categorical and continuous must remain separate if we want to have an embedding layer!
        """
        self.X_cont = cat((self.X_cont, self.X_cat), 1)
        self.X_cat, self.encoding_sizes = None, None
        self.getter = self.define_getter(None)


def load_warmup_dataset(dm, split=True, add_biases=False):
    """
    Loads 'Learning_0_6MWT_and_Generals (WarmUp)' table and create a PetaleDataset object

    :param dm: PetaleDataManager
    :param split: Boolean indicating if we want to split categorical variables from the continuous ones
    :param add_biases: boolean indicating if a column of ones should be added at the beginning of X_cont
    :return: PetaleDataset
    """
    # We save some important constants
    CONTINUOUS_COL = ['34503 Weight', '35149 TDM6_HR_6_2', '35142 TDM6_Distance_2',
                      'Duration of treatment', 'Age', 'MVLPA']
    ID = "Participant"
    TARGET = '35009 EE_VO2r_max'

    df = dm.get_table('Learning_0_6MWT_and_Generals (WarmUp)')

    return PetaleDataset(df, CONTINUOUS_COL, TARGET, ID, split=split, add_biases=add_biases)
