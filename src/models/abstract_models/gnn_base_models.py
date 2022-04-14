"""
Filename: gnn_base_models.py

Author: Nicolas Raymond

Description: This file defines the GNN class which implements common training routine
             methods for GNN models

Date of last modification: 2022/04/13
"""

from src.data.processing.gnn_datasets import MaskType, PetaleKGNNDataset
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import Metric
from torch import no_grad, ones, tensor
from torch.nn import BatchNorm1d, Dropout, Linear
from torch.utils.data import DataLoader
from typing import Callable, List, Optional, Union, Tuple


class GNN(TorchCustomModel):
    """
    Graph Neural Network abstract model
    """
    def __init__(self,
                 output_size: int,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 alpha: float = 0,
                 beta: float = 0,
                 hidden_size: Optional[int] = None,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False):

        """
       Sets some protected attributes

        Args:
            criterion: loss function of our model
            criterion_name: name of the loss function
            eval_metric: evaluation metric
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            hidden_size: size of the hidden states after the graph convolution
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            verbose: True if we want trace of the training progress
        """
        # We call parent's constructor
        super().__init__(criterion=criterion,
                         criterion_name=criterion_name,
                         eval_metric=eval_metric,
                         output_size=output_size,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         verbose=verbose)

        # We save the hidden size
        self._hidden_size = hidden_size if hidden_size is not None else self._input_size

        # We save the batch norm layer
        self._bn = BatchNorm1d(self._hidden_size + self._input_size)

        # We save the dropout layer
        self._dropout = Dropout(0.25)

        # We save the linear layer for the final output
        self._linear_layer = Linear(self._hidden_size + self._input_size, output_size)

    def _execute_train_step(self, train_data: Tuple[DataLoader, PetaleKGNNDataset]) -> float:
        """
        Executes one training epoch

        Args:
            train_data: tuple (train loader, dataset)

        Returns: mean epoch loss
        """

        # We set the model for training
        self.train()
        epoch_loss, epoch_score = 0, 0

        # We extract the training dataloader and the complete dataset
        train_loader, dataset = train_data

        # We extract train_subgraph, train_mask and train_idx_map
        train_subgraph, train_idx_map, train_mask = dataset.train_subgraph

        # We extract the features related to all the train mask
        x, _, _ = dataset[train_mask]

        # We execute one training step
        for item in train_loader:

            # We extract the data
            _, y, idx = item

            # We map the original idx to their position in the train mask
            pos_idx = [train_idx_map[i.item()] for i in idx]

            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the weight update
            pred, loss = self._update_weights([train_subgraph, x], y, pos_idx)

            # We update the metrics history
            score = self._eval_metric(pred, y)
            epoch_loss += loss
            epoch_score += score

        # We update evaluations history
        mean_epoch_loss = self._update_evaluations_progress(epoch_loss, epoch_score,
                                                            nb_batch=len(train_data),
                                                            mask_type=MaskType.TRAIN)

        return mean_epoch_loss

    def _execute_valid_step(self,
                            valid_data: Optional[Union[DataLoader, Tuple[DataLoader, PetaleKGNNDataset]]],
                            early_stopper: Optional[EarlyStopper]) -> bool:
        """
        Executes an inference step on the validation data and apply early stopping if needed

        Args:
            valid_data: tuple (valid loader, dataset)
            early_stopper: early stopper keeping track of validation loss

        Returns: True if we need to early stop
        """
        # We check if there is validation to do
        if valid_data is None:
            return False

        # We extract train loader, dataset
        valid_loader, dataset = valid_data

        # We extract valid_subgraph, mask (train + valid) and valid_idx_map
        valid_subgraph, valid_idx_map, mask = dataset.valid_subgraph

        # We extract the features related to all the train + valid
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()
        epoch_loss, epoch_score = 0, 0

        # We execute one inference step on validation set
        with no_grad():

            for item in valid_loader:

                # We extract the data
                _, y, idx = item

                # We map original idx to their position in the train mask
                pos_idx = [valid_idx_map[i.item()] for i in idx]

                # We perform the forward pass
                pred = self(valid_subgraph, x)

                # We calculate the loss and the score
                epoch_loss += self.loss(pred[pos_idx], y).item()
                epoch_score += self._eval_metric(pred[pos_idx], y)

        # We update evaluations history
        mean_epoch_score = self._update_evaluations_progress(epoch_loss, epoch_score,
                                                             nb_batch=len(valid_loader),
                                                             mask_type=MaskType.VALID)
        # We check early stopping status
        early_stopper(mean_epoch_score, self)

        if early_stopper.early_stop:
            return True

        return False
