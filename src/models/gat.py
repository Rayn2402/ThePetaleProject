"""
Filename: gat.py

Author: Nicolas Raymond

Description: This file is used to define the wrapper for the GATRegressor

Date of last modification: 2022/03/25
"""

from src.models.abstract_models.gat_base_models import GATRegressor
from src.models.wrappers.torch_wrappers import TorchRegressorWrapper
from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP
from src.utils.score_metrics import RegressionMetric
from typing import List, Optional


class PetaleGATR(TorchRegressorWrapper):
    """
    Graph Attention Network regression model wrapper for the Petale framework
    """
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 eval_metric: Optional[RegressionMetric] = None,
                 feat_dropout: float = 0,
                 attn_dropout: float = 0,
                 lr: float = 0.05,
                 rho: float = 0,
                 max_epochs: int = 200,
                 patience: int = 15,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):
        """
        Creates the regression model and sets protected attributes using parent's constructor

        Args:
            hidden_size: size of the hidden states after the graph convolution
            num_heads: number of attention heads
            eval_metric: evaluation metric
            feat_dropout: features dropout probability
            attention dropout probability
            lr: learning rate
            rho: if >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, standard Adam optimizer will be used
            max_epochs: maximum number of epochs for training
            patience: number of consecutive epochs without improvement
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col:
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            verbose: if True, training progress will be printed
        """

        # Creation of model
        model = GATRegressor(hidden_size=hidden_size,
                             num_heads=num_heads,
                             eval_metric=eval_metric,
                             feat_dropout=feat_dropout,
                             attn_dropout=attn_dropout,
                             alpha=alpha,
                             beta=beta,
                             num_cont_col=num_cont_col,
                             cat_idx=cat_idx,
                             cat_sizes=cat_sizes,
                             cat_emb_sizes=cat_emb_sizes,
                             verbose=verbose)

        super().__init__(model=model,
                         train_params={'lr': lr,
                                       'rho': rho,
                                       'batch_size': None,
                                       'valid_batch_size': None,
                                       'patience': patience,
                                       'max_epochs': max_epochs})

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(GATHP())


class GATHP:
    """
    GAT's hyperparameters
    """
    ALPHA = NumericalContinuousHP("alpha")
    ATTN_DROPOUT = NumericalContinuousHP("attn_dropout")
    BETA = NumericalContinuousHP("beta")
    FEAT_DROPOUT = NumericalContinuousHP("feat_dropout")
    HIDDEN_SIZE = NumericalIntHP("hidden_size")
    LR = NumericalContinuousHP("lr")
    NUM_HEADS = NumericalIntHP("num_heads")
    RHO = NumericalContinuousHP("rho")
    WEIGHT = NumericalContinuousHP("weight")

    def __iter__(self):
        return iter([self.ALPHA, self.ATTN_DROPOUT, self.BETA, self.FEAT_DROPOUT,
                     self.HIDDEN_SIZE, self.LR, self.NUM_HEADS, self.RHO])