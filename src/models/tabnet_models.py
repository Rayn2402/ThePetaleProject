"""
Author: Nicolas Raymond

This file is used to implement wrappers for TabNet regression and classification models
"""
from numpy import array
from torch import tensor

from src.models.base_models import PetaleBinaryClassifier, PetaleRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from typing import Optional, List, Union


class PetaleTNC(PetaleBinaryClassifier):
    """
    Class used as a wrapper for TabNet classifier model
    """
    def __init__(self, n_d: int = 8, n_a: int = 8, n_steps: int = 3, gamma: float = 1.3, lr: float = 0.1,
                 cat_idxs: Optional[List[int]] = None, cat_dims: Optional[List[int]] = None,
                 cat_emb_dim: Optional[List[int]] = None, device='cpu', verbose: bool = False,
                 classification_threshold: float = 0.5, weight: Optional[float] = None):
        """
        Creates a TabNet classifier and sets protected attributes using parent's constructor

        Args:
            n_d: Width of the decision prediction layer. Bigger values gives more capacity to the model
                 with the risk of overfitting. Values typically range from 8 to 64.
            n_a: Width of the attention embedding for each mask. According to the paper
                 n_d=n_a is usually a good choice. (default=8)
            n_steps: Number of steps in the architecture (usually between 3 and 10)
            gamma: This is the coefficient for feature reusage in the masks. A value close to 1 will make
                   mask selection least correlated between layers. Values range from 1.0 to 2.0.
            lr: learning rate
            cat_idxs: List of categorical features indices.
            cat_dims: List of categorical features number of modalities
            cat_emb_dim: List of embeddings size for each categorical features.
            device: 'cpu' or 'gpu'
            verbose: True to show training loss progression
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        if cat_idxs is None:
            cat_idxs, cat_dims, cat_emb_dim = [], [], 1

        self.__model = TabNetClassifier(n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                                        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
                                        device_name=device, optimizer_params=dict(lr=lr), verbose=int(verbose))

        # Call of parent's constructor
        super().__init__(classification_threshold=classification_threshold, weight=weight)

    def fit(self, x_train: array, y_train: array, **kwargs) -> None:
        """
        Fits the model to the training data

        Args:
            x_train: (N,D) array with D-dimensional samples
            y_train: (N,1) array with classification labels
            **kwargs: 'batch_size': Number of examples per batch. Large batch sizes are recommended. (default =1024)
                      'eval_set' : List with tuple containing valid set
                                   for early stopping (x_valid, y_valid) (default = None)
                      'max_epochs': Maximum number of epochs for training. (default = 200)
                      'patience': Number of consecutive epochs without improvement (default = 15)

        Returns: None
        """
        sample_weights = self.get_sample_weights(y_train)

        self.__model.fit(x_train, y_train, weights=sample_weights, **kwargs)

    def predict_proba(self, x: array) -> array:
        """
        Returns the probabilities of being in class 1 for all samples

        Args:
            x: (N,D) tensor or array with D-dimensional samples

        Returns: (N,) array
        """
        return self.__model.predict_proba(x)[:, 1]


class PetaleTNR(PetaleRegressor):
    """
    Class used as a wrapper for TabNet regression model
    """

    def __init__(self, n_d: int = 8, n_a: int = 8, n_steps: int = 3, gamma: float = 1.3, lr: float = 0.1,
                 cat_idxs: Optional[List[int]] = None, cat_dims: Optional[List[int]] = None,
                 cat_emb_dim: Optional[List[int]] = None, device='cpu', verbose: bool = False):
        """
        Creates a TabNet classifier and sets protected attributes using parent's constructor

        Args:
            n_d: Width of the decision prediction layer. Bigger values gives more capacity to the model
                 with the risk of overfitting. Values typically range from 8 to 64.
            n_a: Width of the attention embedding for each mask. According to the paper
                 n_d=n_a is usually a good choice. (default=8)
            n_steps: Number of steps in the architecture (usually between 3 and 10)
            gamma: This is the coefficient for feature reusage in the masks. A value close to 1 will make
                   mask selection least correlated between layers. Values range from 1.0 to 2.0.
            lr: learning rate
            cat_idxs: List of categorical features indices.
            cat_dims: List of categorical features number of modalities
            cat_emb_dim: List of embeddings size for each categorical features.
            device: 'cpu' or 'gpu'
            verbose: True to show training loss progression
        """
        if cat_idxs is None:
            cat_idxs, cat_dims, cat_emb_dim = [], [], 1

        self._model = TabNetRegressor(n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                                      cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
                                      device_name=device, optimizer_params=dict(lr=lr), verbose=int(verbose))

    def fit(self, x_train: array, y_train: array, **kwargs) -> None:
        """
        Fits the model to the training data

        Args:
            x_train: (N,D) array with D-dimensional samples
            y_train: (N,1) array with classification labels
            **kwargs: 'batch_size': Number of examples per batch. Large batch sizes are recommended. (default =1024)
                      'eval_set' : List with tuple containing valid set
                                   for early stopping (x_valid, y_valid) (default = None)
                      'max_epochs': Maximum number of epochs for training. (default = 200)
                      'patience': Number of consecutive epochs without improvement (default = 15)

        Returns: None
        """
        self._model.fit(x_train, y_train.reshape(-1, 1), **kwargs)

    def predict(self, x: array) -> array:
        """
        Returns the predicted real-valued targets for all samples

        Args:
            x: (N,D) array with D-dimensional samples

        Returns: (N,) array
        """
        return self._model.predict(x).squeeze()
