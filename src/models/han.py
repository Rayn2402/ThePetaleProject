"""
Author: Nicolas Raymond

This file is use to store the regression and classification wrappers for HAN models
"""
from src.models.torch_wrappers import TorchBinaryClassifierWrapper
from src.models.abstract_models.han_base_models import HANBinaryClassifier
from src.utils.score_metrics import BinaryClassificationMetric
from typing import List, Optional


class PetaleBinaryHANC(TorchBinaryClassifierWrapper):
    """
    Heterogeneous graph attention binary classification network adapted to Petale framework
    """
    def __init__(self, meta_paths: List[List[str]], in_size: int, hidden_size: int,
                 num_heads: int, dropout: float = 0, lr: float = 0.05, batch_size: int = 55,
                 valid_batch_size: Optional[int] = None, max_epochs: int = 200, patience: int = 15,
                 eval_metric: Optional[BinaryClassificationMetric] = None, alpha: float = 0, beta: float = 0,
                 classification_threshold: float = 0.5, weight:  Optional[float] = None, verbose: bool = False):
        """
        Creates the classifier and sets protected attributes using parent's constructor

        Args:
            meta_paths: List of metapaths, each meta path is a list of edge types
            in_size: input size (number of features per node)
            hidden_size: size of embedding learnt within each attention head
            num_heads: int representing the number of attention heads
            dropout: dropout probability
            lr: learning rate
            batch_size: size of the batches in the training loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: Maximum number of epochs for training
            patience: Number of consecutive epochs without improvement
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            classification_threshold: threshold used to classify a sample in class 1
            weight: weight attributed to class 1
            verbose: True if we want to show the training progress
        """
        # Creation of model
        model = HANBinaryClassifier(meta_paths=meta_paths, in_size=in_size, hidden_size=hidden_size,
                                    num_heads=num_heads, dropout=dropout, eval_metric=eval_metric,
                                    alpha=alpha, beta=beta, verbose=verbose)

        super().__init__(model=model, classification_threshold=classification_threshold, weight=weight,
                         train_params={'lr': lr, 'batch_size': batch_size, 'valid_batch_size': valid_batch_size,
                                       'patience': patience, 'max_epochs': max_epochs})