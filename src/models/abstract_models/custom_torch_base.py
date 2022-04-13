"""
Filename: custom_torch_base.py

Author: Nicolas Raymond

Description: Defines the abstract class TorchCustomModel from which all custom pytorch models
             implemented for the project must inherit. This class allows to store common
             function of all pytorch models.

Date of last modification: 2022/04/13

"""

from abc import ABC, abstractmethod
from dgl import DGLGraph
from src.data.processing.datasets import MaskType, PetaleDataset
from src.data.processing.gnn_datasets import PetaleKGNNDataset
from src.models.blocks.mlp_blocks import EntityEmbeddingBlock
from src.training.early_stopping import EarlyStopper
from src.training.sam import SAM
from src.utils.score_metrics import Metric
from src.utils.visualization import visualize_epoch_progression
from torch import tensor
from torch.nn import BatchNorm1d, Module
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Any, Callable, List, Optional, Tuple, Union


class TorchCustomModel(Module, ABC):
    """
    Abstract class used to store common attributes
    and methods of torch models implemented in the project
    """
    def __init__(self,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 output_size: int,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 additional_input_args: Optional[List[Any]] = None,
                 verbose: bool = False):
        """
        Sets the protected attributes and creates an embedding block if required

        Args:
            criterion: loss function of our model
            criterion_name: name of the loss function
            eval_metric: evaluation metric of our model (Ex. accuracy, mean absolute error)
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns in the dataset,
                          cont idx are assumed to be range(num_cont_col)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            additional_input_args: list of arguments that must be also considered when validating
                                   input arguments
            verbose: true if we want to print the training progress
        """

        # We validate input arguments (check if there are continuous or categorical inputs)
        additional_input_args = additional_input_args if additional_input_args is not None else []
        self._validate_input_args([num_cont_col, cat_sizes, *additional_input_args])

        # Call of parent's constructor
        Module.__init__(self)

        # Settings of general protected attributes
        self._alpha = alpha
        self._beta = beta
        self._criterion = criterion
        self._criterion_name = criterion_name
        self._eval_metric = eval_metric
        self._evaluations = {i: {self._criterion_name: [],
                                 self._eval_metric.name: []} for i in [MaskType.TRAIN, MaskType.VALID]}
        self._input_size = num_cont_col if num_cont_col is not None else 0
        self._optimizer = None
        self._output_size = output_size
        self._verbose = verbose

        # Settings of protected attributes related to entity embedding
        self._cat_idx = cat_idx if cat_idx is not None else []
        self._cont_idx = list(range(num_cont_col))
        self._embedding_block = None

        # Initialization of a protected method
        self._update_weights = None

        # We set the embedding layers
        if len(cat_idx) != 0 and cat_sizes is not None:

            # We check embedding sizes (if nothing provided -> emb_sizes = cat_sizes - 1)
            if cat_emb_sizes is None:
                cat_emb_sizes = [s - 1 for s in cat_sizes]
                if 0 in cat_emb_sizes:
                    raise ValueError('One categorical variable as a single modality')

            # We create the embedding layers
            self._embedding_block = EntityEmbeddingBlock(cat_sizes, cat_emb_sizes, cat_idx)

            # We sum the length of all embeddings
            self._input_size += self._embedding_block.output_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def _create_validation_objects(self,
                                   dataset: PetaleDataset,
                                   valid_batch_size: Optional[int],
                                   patience: int
                                   ) -> Tuple[EarlyStopper, DataLoader]:
        """
        Creates the objects needed for validation during the training process

        Args:
            dataset: PetaleDataset used to feed the dataloader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            patience: number of consecutive epochs without improvement allowed

        Returns: EarlyStopper, Dataloader

        """
        # We create the valid dataloader (if valid size != 0)
        valid_size, valid_data, early_stopper = len(dataset.valid_mask), None, None

        if valid_size != 0:

            # We check if a valid batch size was provided
            valid_batch_size = min(valid_size, valid_batch_size) if valid_batch_size is not None else valid_size

            # We create the valid loader
            valid_data = DataLoader(dataset,
                                    batch_size=valid_batch_size,
                                    sampler=SubsetRandomSampler(dataset.valid_mask))

            # We create the early stopper
            early_stopper = EarlyStopper(patience, self._eval_metric.direction)

        return early_stopper, valid_data

    def _disable_running_stats(self) -> None:
        """
        Disables batch norm momentum when executing SAM optimization step

        Returns: None
        """
        self.apply(self.disable_module_running_stats)

    def _enable_running_stats(self) -> None:
        """
        Restores batch norm momentum when executing SAM optimization step

        Returns: None
        """
        self.apply(self.enable_module_running_stats)

    def _sam_weight_update(self,
                           x: List[Union[DGLGraph, tensor]],
                           y: tensor,
                           pos_idx: Optional[List[int]] = None) -> Tuple[tensor, float]:
        """
        Executes a weights update using Sharpness-Aware Minimization (SAM) optimizer

        Note from https://github.com/davda54/sam :
            The running statistics are computed in both forward passes, but they should
            be computed only for the first one. A possible solution is to set BN momentum
            to zero to bypass the running statistics during the second pass.

        Args:
            x: list of arguments taken for the forward pass (DGLGraph and (N', D) tensor with batch inputs)
            y: (N',) ground truth associated to a batch
            pos_idx: dictionary that maps the original dataset's idx to their current
                     position in the mask used for the forward pass (used only with GNNs)

        Returns: (N',) tensor with predictions, training loss
        """
        # We compute the predictions
        pred = self(*x)
        pred = pred if pos_idx is None else pred[pos_idx]

        # First forward-backward pass
        loss = self.loss(pred, y)
        loss.backward()
        self._optimizer.first_step()

        # Second forward-backward pass
        self._disable_running_stats()
        second_pred = self(*x)
        second_pred = second_pred if pos_idx is None else second_pred[pos_idx]
        self.loss(second_pred, y).backward()
        self._optimizer.second_step()

        # We enable running stats again
        self._enable_running_stats()

        return pred, loss.item()

    def _basic_weight_update(self,
                             x: List[Union[DGLGraph, tensor]],
                             y: tensor,
                             pos_idx: Optional[List[int]] = None) -> Tuple[tensor, float]:
        """
        Executes a weights update without using Sharpness-Aware Minimization (SAM)

        Args:
            x: list of arguments taken for the forward pass (DGLGraph and (N', D) tensor with batch inputs)
            y: (N',) ground truth associated to a batch
            pos_idx: dictionary that maps the original dataset's idx to their current
                     position in the mask used for the forward pass (used only with GNNs)

        Returns: (N',) tensor with predictions, training loss
        """
        # We compute the predictions
        pred = self(*x)
        pred = pred if pos_idx is None else pred[pos_idx]

        # We execute a single forward-backward pass
        loss = self.loss(pred, y)
        loss.backward()
        self._optimizer.step()

        return pred, loss.item()

    def _generate_progress_func(self, max_epochs: int) -> Callable:
        """
        Builds a function that updates the training progress in the terminal

        Args:
            max_epochs: maximum number of training epochs

        Returns: function
        """
        if self._verbose:
            def update_progress(epoch: int, mean_epoch_loss: float):
                if (epoch + 1) % 5 == 0 or (epoch + 1) == max_epochs:
                    print(f"Epoch {epoch + 1} - Loss : {round(mean_epoch_loss, 4)}")
        else:
            def update_progress(*args):
                pass

        return update_progress

    def print_early_stopping_message(self,
                                     epoch: int,
                                     patience: int,
                                     best_validation_score: float) -> None:
        """
        Prints a message when early stopping occurs

        Args:
            epoch: number of training epochs done
            patience: number of consecutive epochs without improvement allowed
            best_validation_score: best validation score obtained

        Returns: None
        """
        print(f"\nEarly stopping occurred at epoch {epoch} with best_epoch = {epoch - patience}"
              f" and best_val_{self._eval_metric.name} = {round(best_validation_score, 4)}")

    def _update_evaluations_progress(self,
                                     loss: float,
                                     score: float,
                                     nb_batch: int,
                                     mask_type: str) -> float:
        """
        Add epoch score and loss to the evaluations history

        Args:
            loss: epoch loss
            score: epoch score
            nb_batch: nb of batches
            mask_type: "train" of "valid"

        Returns: mean epoch loss or mean epoch score
        """

        # We compute the loss average per batch
        loss /= nb_batch
        score /= nb_batch

        # We update the evaluations history
        self._evaluations[mask_type][self._criterion_name].append(loss)
        self._evaluations[mask_type][self._eval_metric.name].append(score)

        # We return a value according to the mask type
        if mask_type == MaskType.VALID:
            return score

        return loss

    def fit(self,
            dataset: PetaleDataset,
            lr: float,
            rho: float = 0,
            batch_size: Optional[int] = 55,
            valid_batch_size: Optional[int] = None,
            max_epochs: int = 200,
            patience: int = 15) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDataset used to feed the dataloaders
            lr: learning rate
            rho: if >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, Adam optimizer will be used
            batch_size: size of the batches in the training loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: Maximum number of epochs for training
            patience: Number of consecutive epochs without improvement allowed

        Returns: None
        """
        # We create the training objects
        train_data = self._create_train_dataloader(dataset, batch_size)

        # We create the objects needed for validation (data loader, early stopper)
        early_stopper, valid_data = self._create_validation_objects(dataset, valid_batch_size, patience)

        # We init the update function
        update_progress = self._generate_progress_func(max_epochs)

        # We set the optimizer
        if rho > 0:
            self._update_weights = self._sam_weight_update
            self._optimizer = SAM(self.parameters(), Adam, rho=rho, lr=lr)
        else:
            self._update_weights = self._basic_weight_update
            self._optimizer = Adam(self.parameters(), lr=lr)

        # We add the dataset to train_data and valid_data if it is a GNN dataset
        if isinstance(dataset, PetaleKGNNDataset):
            train_data = (train_data, dataset)
            valid_data = (valid_data, dataset)

        # We execute the epochs
        for epoch in range(max_epochs):

            # We calculate training loss
            train_loss = self._execute_train_step(train_data)
            update_progress(epoch, train_loss)

            # We calculate valid score and apply early stopping if needed
            if self._execute_valid_step(valid_data, early_stopper):
                self.print_early_stopping_message(epoch, patience, early_stopper.best_val_score)
                break

        if early_stopper is not None:

            # We extract best params and remove checkpoint file
            self.load_state_dict(early_stopper.get_best_params())
            early_stopper.remove_checkpoint()

    def loss(self,
             pred: tensor,
             y: tensor) -> tensor:
        """
        Calls the criterion and add the elastic penalty

        Args:
            pred: (N, C) tensor if classification with C classes, (N,) tensor for regression
            y: (N,) tensor with targets

        Returns: tensor with loss value
        """
        # Computations of penalties
        l1_penalty, l2_penalty = tensor(0.), tensor(0.)
        for _, w in self.named_parameters():
            l1_penalty = l1_penalty + w.abs().sum()
            l2_penalty = l2_penalty + w.pow(2).sum()

        # Computation of loss reduction + elastic penalty
        return self._criterion(pred, y.float()) + self._alpha * l1_penalty + self._beta * l2_penalty

    def plot_evaluations(self, save_path: Optional[str] = None) -> None:
        """
        Plots the training and valid curves saved

        Args:
            save_path: path were the figures will be saved

        Returns: None
        """
        # Extraction of data
        train_loss = self._evaluations[MaskType.TRAIN][self._criterion_name]
        train_metric = self._evaluations[MaskType.TRAIN][self._eval_metric.name]
        valid_loss = self._evaluations[MaskType.VALID][self._criterion_name]
        valid_metric = self._evaluations[MaskType.VALID][self._eval_metric.name]

        # Figure construction
        visualize_epoch_progression(train_history=[train_loss, train_metric],
                                    valid_history=[valid_loss, valid_metric],
                                    progression_type=[self._criterion_name, self._eval_metric.name],
                                    path=save_path)

    @staticmethod
    def _create_train_dataloader(dataset: PetaleDataset,
                                 batch_size: int
                                 ) -> DataLoader:
        """
        Creates the objects needed for the training

        Args:
            dataset: PetaleDataset used to feed the dataloaders
            batch_size: size of the batches in the train loader

        Returns: train loader

        """
        # Creation of training loader
        train_size = len(dataset.train_mask)
        batch_size = min(train_size, batch_size) if batch_size is not None else train_size
        train_data = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(dataset.train_mask),
                                drop_last=(train_size % batch_size) == 1)

        return train_data

    @staticmethod
    def _validate_input_args(input_args: List[Any]) -> None:
        """
        Checks if all arguments related to inputs are None,
        if not the inputs are valid

        Args:
            input_args: list of arguments related to inputs

        Returns: None
        """
        valid = False
        for arg in input_args:
            if arg is not None:
                valid = True
                break

        if not valid:
            raise ValueError("There must be continuous columns or categorical columns")

    @abstractmethod
    def _execute_train_step(self, train_data: Union[DataLoader, Tuple[DataLoader, PetaleDataset]]) -> float:
        """
        Executes one training epoch

        Args:
            train_data: training dataloader or tuple (train loader, dataset)

        Returns: mean epoch loss
        """
        raise NotImplementedError

    @abstractmethod
    def _execute_valid_step(self,
                            valid_data: Optional[Union[DataLoader, Tuple[DataLoader, PetaleDataset]]],
                            early_stopper: EarlyStopper) -> bool:
        """
        Executes an inference step on the validation data

        Args:
            valid_data: valid dataloader or tuple (valid loader, dataset)

        Returns: True if we need to early stop
        """
        raise NotImplementedError

    @staticmethod
    def disable_module_running_stats(module: Module) -> None:
        """
        Sets momentum to 0 for all BatchNorm layer in the module after saving it in a cache

        Args:
            module: torch module

        Returns: None
        """
        if isinstance(module, BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    @staticmethod
    def enable_module_running_stats(module: Module) -> None:
        """
        Restores momentum for all BatchNorm layer in the module using the value in the cache

        Args:
            module: torch module

        Returns: None
        """
        if isinstance(module, BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
