"""
Filename: mlp.py

Authors: Nicolas Raymond

Description: This file is used to define the regression and classification
             wrappers for MLP models

Date of last modification : 2021/10/25
"""

from src.models.wrappers.torch_wrappers import TorchBinaryClassifierWrapper, TorchRegressorWrapper
from src.models.abstract_models.mlp_base_models import MLPBinaryClassifier, MLPRegressor
from src.utils.hyperparameters import CategoricalHP, HP, NumericalContinuousHP, NumericalIntHP
from src.utils.score_metrics import Metric, BinaryClassificationMetric
from typing import List, Optional


class PetaleBinaryMLPC(TorchBinaryClassifierWrapper):
    """
    Multilayer perceptron classification model with entity embedding adapted to Petale framework
    """
    def __init__(self,
                 n_layer: int,
                 n_unit: int,
                 activation: str,
                 eval_metric: Optional[BinaryClassificationMetric] = None,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 lr: float = 0.05,
                 batch_size: int = 55,
                 valid_batch_size: Optional[int] = None,
                 max_epochs: int = 200,
                 patience: int = 15,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False,
                 classification_threshold: float = 0.5,
                 weight:  Optional[float] = None):
        """
        Builds a binary classification MLP and sets the protected attributes using parent's constructor

        Args:
            n_layer: number of hidden layer
            n_unit: number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            lr: learning rate
            batch_size: size of the batches in the training loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: maximum number of epochs for training
            patience: number of consecutive epochs without improvement
            num_cont_col: number of numerical continuous columns
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """
        # Model creation
        model = MLPBinaryClassifier(layers=[n_unit]*n_layer,
                                    activation=activation,
                                    eval_metric=eval_metric,
                                    dropout=dropout,
                                    alpha=alpha,
                                    beta=beta,
                                    num_cont_col=num_cont_col,
                                    cat_idx=cat_idx,
                                    cat_sizes=cat_sizes,
                                    cat_emb_sizes=cat_emb_sizes,
                                    verbose=verbose)

        super().__init__(model=model,
                         classification_threshold=classification_threshold,
                         weight=weight,
                         train_params={'lr': lr,
                                       'batch_size': batch_size,
                                       'valid_batch_size': valid_batch_size,
                                       'patience': patience,
                                       'max_epochs': max_epochs})

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(MLPHP()) + [MLPHP.WEIGHT]


class PetaleMLPR(TorchRegressorWrapper):
    """
    Multilayer perceptron regression model with entity embedding adapted to Petale framework
    """
    def __init__(self,
                 n_layer: int,
                 n_unit: int,
                 activation: str,
                 eval_metric: Optional[Metric] = None,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 lr: float = 0.05,
                 batch_size: int = 55,
                 valid_batch_size: Optional[int] = None,
                 max_epochs: int = 200,
                 patience: int = 15,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):
        """
        Builds and MLP regression model and sets the protected attributes using parent's constructor

        Args:
            n_layer: number of hidden layer
            n_unit: number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            lr: learning rate
            batch_size: size of the batches in the training loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: maximum number of epochs for training
            patience: number of consecutive epochs without improvement
            num_cont_col: number of numerical continuous columns
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
        """
        # Creation of the model
        model = MLPRegressor(layers=[n_unit]*n_layer,
                             activation=activation,
                             eval_metric=eval_metric,
                             dropout=dropout,
                             alpha=alpha,
                             beta=beta,
                             num_cont_col=num_cont_col,
                             cat_idx=cat_idx,
                             cat_sizes=cat_sizes,
                             cat_emb_sizes=cat_emb_sizes,
                             verbose=verbose)

        # Call of parent's constructor
        super().__init__(model=model,
                         train_params={'lr': lr,
                                       'batch_size': batch_size,
                                       'valid_batch_size': valid_batch_size,
                                       'patience': patience,
                                       'max_epochs': max_epochs})

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(MLPHP())


class MLPHP:
    """
    MLP's hyperparameters
    """
    ACTIVATION = CategoricalHP("activation")
    ALPHA = NumericalContinuousHP("alpha")
    BATCH_SIZE = NumericalIntHP("batch_size")
    BETA = NumericalContinuousHP("beta")
    LR = NumericalContinuousHP("lr")
    N_LAYER = NumericalIntHP("n_layer")
    N_UNIT = NumericalIntHP("n_unit")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.ACTIVATION, self.ALPHA, self.BATCH_SIZE, self.BETA, self.LR,
                     self.N_LAYER, self.N_UNIT])
