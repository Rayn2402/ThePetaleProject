"""
Filename: han.py

Authors: Nicolas Raymond

Description: This file is used to define the regression and classification wrappers for HAN models

Date of last modification : 2021/11/25
"""

from src.models.abstract_models.han_base_models import HANBinaryClassifier, HANRegressor
from src.models.wrappers.torch_wrappers import TorchBinaryClassifierWrapper, TorchRegressorWrapper
from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP
from src.utils.score_metrics import BinaryClassificationMetric
from typing import Callable, List, Optional


class PetaleBinaryHANC(TorchBinaryClassifierWrapper):
    """
    Heterogeneous graph attention binary classification network wrapper for the Petale framework
    """
    def __init__(self,
                 meta_paths: List[List[str]],
                 hidden_size: int,
                 num_heads: int,
                 cat_idx: List[int],
                 cat_sizes: List[int],
                 cat_emb_sizes: List[int],
                 num_cont_col: Optional[int] = None,
                 dropout: float = 0,
                 lr: float = 0.05,
                 rho: float = 0,
                 batch_size: int = 55,
                 valid_batch_size: Optional[int] = None,
                 max_epochs: int = 200,
                 patience: int = 15,
                 eval_metric: Optional[BinaryClassificationMetric] = None,
                 alpha: float = 0,
                 beta: float = 0,
                 pre_encoder_constructor: Callable = None,
                 classification_threshold: float = 0.5,
                 weight:  Optional[float] = None,
                 verbose: bool = False):
        """
        Creates the classifier and sets protected attributes using parent's constructor

        Args:
            meta_paths: list of metapaths, each meta path is a list of edge types
            hidden_size: size of embedding learnt within each attention head
            num_heads: number of attention heads
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            dropout: dropout probability
            lr: learning rate
            rho: if >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, standard Adam optimizer will be used
            batch_size: size of the batches in the training loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: maximum number of epochs for training
            patience: number of consecutive epochs without improvement
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            pre_encoder_constructor: function that creates an encoder that goes after the entity embedding block
                                     This function must have a parameter "input_size"
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
            verbose: True if we want to show the training progress
        """
        # Creation of model
        model = HANBinaryClassifier(meta_paths=meta_paths,
                                    hidden_size=hidden_size,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    num_cont_col=num_cont_col,
                                    cat_idx=cat_idx,
                                    cat_sizes=cat_sizes,
                                    cat_emb_sizes=cat_emb_sizes,
                                    eval_metric=eval_metric,
                                    alpha=alpha,
                                    beta=beta,
                                    pre_encoder_constructor=pre_encoder_constructor,
                                    verbose=verbose)

        super().__init__(model=model,
                         classification_threshold=classification_threshold,
                         weight=weight,
                         train_params={'lr': lr,
                                       'rho': rho,
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
        return list(HanHP()) + [HanHP.WEIGHT]


class PetaleHANR(TorchRegressorWrapper):
    """
    Heterogeneous graph attention regression network wrapper for the Petale framework
    """
    def __init__(self,
                 meta_paths: List[List[str]],
                 hidden_size: int,
                 num_heads: int,
                 cat_idx: List[int],
                 cat_sizes: List[int],
                 cat_emb_sizes: List[int],
                 num_cont_col: Optional[int] = None,
                 dropout: float = 0,
                 lr: float = 0.05,
                 rho: float = 0,
                 batch_size: int = 55,
                 valid_batch_size: Optional[int] = None,
                 max_epochs: int = 200,
                 patience: int = 15,
                 eval_metric: Optional[BinaryClassificationMetric] = None,
                 alpha: float = 0,
                 beta: float = 0,
                 pre_encoder_constructor: Callable = None,
                 verbose: bool = False):
        """
        Creates the regression model and sets protected attributes using parent's constructor

        Args:
            meta_paths: List of metapaths, each meta path is a list of edge types
            hidden_size: size of embedding learnt within each attention head
            num_heads: number of attention heads
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            dropout: dropout probability
            lr: learning rate
            rho: if >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, standard SGD optimizer with momentum will be used
            batch_size: size of the batches in the training loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: Maximum number of epochs for training
            patience: Number of consecutive epochs without improvement
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            pre_encoder_constructor: function that creates an encoder that goes after the entity embedding block
                                     This function must have a parameter "input_size"
            verbose: True if we want to show the training progress
        """
        # Creation of model
        model = HANRegressor(meta_paths=meta_paths,
                             hidden_size=hidden_size,
                             num_heads=num_heads,
                             dropout=dropout,
                             num_cont_col=num_cont_col,
                             cat_idx=cat_idx,
                             cat_sizes=cat_sizes,
                             cat_emb_sizes=cat_emb_sizes,
                             eval_metric=eval_metric,
                             alpha=alpha,
                             beta=beta,
                             pre_encoder_constructor=pre_encoder_constructor,
                             verbose=verbose)

        super().__init__(model=model,
                         train_params={'lr': lr,
                                       'rho': rho,
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
        return list(HanHP())


class HanHP:
    """
    Han's hyperparameters
    """
    ALPHA = NumericalContinuousHP("alpha")
    BATCH_SIZE = NumericalIntHP("batch_size")
    BETA = NumericalContinuousHP("beta")
    DROPOUT = NumericalContinuousHP("dropout")
    HIDDEN_SIZE = NumericalIntHP("hidden_size")
    LR = NumericalContinuousHP("lr")
    NUM_HEADS = NumericalIntHP("num_heads")
    RHO = NumericalContinuousHP("rho")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.ALPHA, self.BATCH_SIZE, self.BETA, self.DROPOUT,
                     self.HIDDEN_SIZE, self.LR, self.NUM_HEADS, self.RHO])
