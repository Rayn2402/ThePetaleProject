"""
Filename: gcn.py

Author: Nicolas Raymond

Description: This file is used to define the wrappers for the
             GCNClassifier and the GCNRegressor

Date of last modification: 2022/04/13
"""

from src.models.abstract_models.gcn_base_models import GCNClassifier, GCNRegressor
from src.models.wrappers.torch_wrappers import TorchBinaryClassifierWrapper, TorchRegressorWrapper
from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP
from src.utils.score_metrics import RegressionMetric
from typing import List, Optional


class PetaleBinaryGCNC(TorchBinaryClassifierWrapper):
    """
    Graph Attention Network binary classification model wrapper for the Petale framework
    """
    def __init__(self,
                 eval_metric: Optional[RegressionMetric] = None,
                 lr: float = 0.05,
                 rho: float = 0,
                 max_epochs: int = 200,
                 patience: int = 15,
                 alpha: float = 0,
                 beta: float = 0,
                 hidden_size: Optional[int] = None,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False,
                 classification_threshold: float = 0.5,
                 weight:  Optional[float] = None):
        """
        Sets the protected attributes using parent's constructor

        Args:
            eval_metric: evaluation metric
            attention dropout probability
            lr: learning rate
            rho: if >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, standard Adam optimizer will be used
            max_epochs: maximum number of epochs for training
            patience: number of consecutive epochs without improvement
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            hidden_size: size of the hidden states after the graph convolution
            num_cont_col:
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            verbose: if True, training progress will be printed
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
        """

        super().__init__(model_constructor=GCNClassifier,
                         model_params=dict(hidden_size=hidden_size,
                                           eval_metric=eval_metric,
                                           alpha=alpha,
                                           beta=beta,
                                           num_cont_col=num_cont_col,
                                           cat_idx=cat_idx,
                                           cat_sizes=cat_sizes,
                                           cat_emb_sizes=cat_emb_sizes,
                                           verbose=verbose),
                         classification_threshold=classification_threshold,
                         weight=weight,
                         train_params=dict(lr=lr,
                                           rho=rho,
                                           batch_size=None,
                                           valid_batch_size=None,
                                           patience=patience,
                                           max_epochs=max_epochs))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(GCNHP()) + [GCNHP.WEIGHT]


class PetaleGCNR(TorchRegressorWrapper):
    """
    Graph Attention Network regression model wrapper for the Petale framework
    """
    def __init__(self,
                 eval_metric: Optional[RegressionMetric] = None,
                 lr: float = 0.05,
                 rho: float = 0,
                 max_epochs: int = 200,
                 patience: int = 15,
                 alpha: float = 0,
                 beta: float = 0,
                 hidden_size: Optional[int] = None,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):
        """
        Creates the regression model and sets protected attributes using parent's constructor

        Args:
            eval_metric: evaluation metric
            attention dropout probability
            lr: learning rate
            rho: if >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, standard Adam optimizer will be used
            max_epochs: maximum number of epochs for training
            patience: number of consecutive epochs without improvement
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            hidden_size: size of the hidden states after the graph convolution
            num_cont_col:
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            verbose: if True, training progress will be printed
        """

        # Creation of model
        model = GCNRegressor(hidden_size=hidden_size,
                             eval_metric=eval_metric,
                             alpha=alpha,
                             beta=beta,
                             num_cont_col=num_cont_col,
                             cat_idx=cat_idx,
                             cat_sizes=cat_sizes,
                             cat_emb_sizes=cat_emb_sizes,
                             verbose=verbose)

        super().__init__(model=model,
                         train_params=dict(lr=lr,
                                           rho=rho,
                                           batch_size=None,
                                           valid_batch_size=None,
                                           patience=patience,
                                           max_epochs=max_epochs))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(GCNHP())


class GCNHP:
    """
    GCN's hyperparameters
    """
    ALPHA = NumericalContinuousHP("alpha")
    BETA = NumericalContinuousHP("beta")
    HIDDEN_SIZE = NumericalIntHP("hidden_size")
    LR = NumericalContinuousHP("lr")
    RHO = NumericalContinuousHP("rho")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.ALPHA, self.BETA, self.HIDDEN_SIZE,
                     self.LR, self.RHO])