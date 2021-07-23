"""
Author: Nicolas Raymond

This file is use to store the regression and classification wrappers for MLP base model
"""
from src.data.processing.datasets import PetaleDataset
from src.models.base_models import PetaleBinaryClassifier, PetaleRegressor
from src.models.mlp_base_models import MLPBinaryClassifier, MLPRegressor
from src.utils.score_metrics import Metric, BinaryClassificationMetric
from torch import tensor
from typing import List, Optional


class PetaleBinaryMLPC(PetaleBinaryClassifier):
    """
    Multilayer perceptron classification model with entity embedding
    """
    def __init__(self, layers: List[int], activation: str,
                 eval_metric: Optional[BinaryClassificationMetric] = None, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, lr: float = 0.05, num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None, cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None, verbose: bool = False,
                 classification_threshold: float = 0.5, weight:  Optional[float] = None):
        """
        Build a binary classification MLP and sets protected attributes using parent's constructor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            lr: learning rate
            num_cont_col: number of numerical continuous columns
                          (equal to number of class in the case of classification)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        self.__model = MLPBinaryClassifier(layers=layers, activation=activation, eval_metric=eval_metric,
                                           dropout=dropout, alpha=alpha, beta=beta, lr=lr, num_cont_col=num_cont_col,
                                           cat_idx=cat_idx, cat_sizes=cat_sizes, cat_emb_sizes=cat_emb_sizes,
                                           verbose=verbose)

        super().__init__(classification_threshold=classification_threshold, weight=weight)

    def fit(self, dataset: PetaleDataset, **kwargs) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which items are tuples (x, y) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
            kwargs:
                batch_size: int = 55,
                valid_batch_size: Optional[int] = None
                max_epochs: int = 200,
                patience: int = 15

        Returns: None
        """
        # We extract sample weights
        _, y, _ = dataset[list(range(len(dataset)))]
        sample_weights = self.get_sample_weights(y)

        # We run MLP fit method
        self.__model.fit(dataset, sample_weights=sample_weights, **kwargs)

    def predict_proba(self, x: tensor) -> tensor:
        """
        Returns the probabilities of being in class 1 for all samples

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: (N,) tensor
        """
        return self.__model.predict_proba(x)


class PetaleMLPR(PetaleRegressor):
    """
    Class used as a wrapper for MLP regression model
    """
    def __init__(self, layers: List[int], activation: str, eval_metric: Optional[Metric] = None,
                 dropout: float = 0, alpha: float = 0, beta: float = 0, lr: float = 0.05,
                 num_cont_col: Optional[int] = None, cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None, cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):
        """
        Builds and MLP regressor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            lr: learning rate
            num_cont_col: number of numerical continuous columns
                          (equal to number of class in the case of classification)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
        """
        # Creation of the model
        self.__model = MLPRegressor(layers=layers, activation=activation,
                                    eval_metric=eval_metric, dropout=dropout, alpha=alpha, beta=beta,
                                    lr=lr, num_cont_col=num_cont_col, cat_idx=cat_idx, cat_sizes=cat_sizes,
                                    cat_emb_sizes=cat_emb_sizes, verbose=verbose)

    def fit(self, dataset: PetaleDataset, **kwargs) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDatasets which items are tuples (x, y) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
            kwargs:
                batch_size: int = 55,
                valid_batch_size: Optional[int] = None
                max_epochs: int = 200,
                patience: int = 15

        Returns: None
        """
        self.__model.fit(dataset, **kwargs)

    def predict(self, x: tensor) -> tensor:
        """
        Returns the predicted real-valued targets for all samples

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: (N,) tensor
        """
        return self.__model.predict(x)

