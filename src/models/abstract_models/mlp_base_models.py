"""
Filename: mlp_base_models.py

Authors: Nicolas Raymond
         Mehdi Mitiche

Description: This file is used to define the MLP model with entity embeddings
             and its children MLPBinaryClassifier and PetaleMLPRegressor.
             These models are not shaped to inherit from the PetaleRegressor
             and PetaleBinaryClassifier classes. However, two wrapper classes for torch models
             are provided to enable the use of these mlp models with hyperparameter tuning functions.

Date of last modification : 2022/04/13
"""

from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.models.blocks.genes_signature_block import GeneGraphAttentionEncoder
from src.models.blocks.mlp_blocks import MLPEncodingBlock
from src.data.processing.datasets import MaskType, PetaleDataset
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import BinaryCrossEntropy, Metric, RootMeanSquaredError
from torch import cat, no_grad, tensor, ones, sigmoid
from torch.nn import BCEWithLogitsLoss, Dropout, Identity, Linear, MSELoss
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional


class MLP(TorchCustomModel):
    """
    Multilayer perceptron model with entity embedding for categorical variables
    and genomic signature embedding for variables identified as genes.
    """
    def __init__(self,
                 output_size: int,
                 layers: List[int],
                 activation: str,
                 criterion: Callable,
                 criterion_name: str,
                 eval_metric: Metric,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 gene_idx_groups: Optional[Dict[str, List[int]]] = None,
                 gene_encoder_constructor: Optional[Callable] = None,
                 verbose: bool = False):

        """
        Builds the layers of the model and sets other protected attributes

        Args:
            output_size: the number of nodes in the last layer of the neural network
            layers: list with number of units in each hidden layer
            criterion: loss function of our model
            criterion_name: name of the loss function
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns in the dataset
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            gene_encoder_constructor: function that generates a GeneEncoder from gene_idx_groups
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
                         additional_input_args=[gene_idx_groups],
                         verbose=verbose)

        if gene_idx_groups is not None:
            self._genes_encoding_block = gene_encoder_constructor(gene_idx_groups=gene_idx_groups, dropout=dropout)
            self._genes_dropout_layer = Identity() if dropout <= 0 else Dropout(p=dropout)
            self._genes_available = True
            self._input_size += self._genes_encoding_block.output_size
        else:
            self._genes_encoding_block = None
            self._genes_dropout_layer = Identity()
            self._genes_available = False
            self._pre_training = False

        if len(layers) > 0:
            self._main_encoding_block = MLPEncodingBlock(input_size=self._input_size,
                                                         output_size=layers[-1],
                                                         layers=layers[:-1],
                                                         activation=activation,
                                                         dropout=dropout)
        else:
            self._main_encoding_block = Identity()
            layers.append(self._input_size)

        # We add a linear layer to complete the layers
        self._linear_layer = Linear(layers[-1], output_size)

    @property
    def att_dict(self) -> Optional[Dict[str, tensor]]:
        if isinstance(self._genes_encoding_block, GeneGraphAttentionEncoder):
            return self._genes_encoding_block.att_dict
        else:
            return None

    def _execute_train_step(self, train_data: DataLoader) -> float:
        """
        Executes one training epoch

        Args:
            train_data: training dataloader

        Returns: mean epoch loss
        """
        # We set the model for training
        self.train()
        epoch_loss, epoch_score = 0, 0

        # We execute one training step
        for item in train_data:

            # We extract the data
            x, y, _ = item

            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the weight update
            pred, loss = self._update_weights([x], y)

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
                            valid_loader: Optional[DataLoader],
                            early_stopper: EarlyStopper) -> bool:
        """
        Executes an inference step on the validation data

        Args:
            valid_loader: validation dataloader

        Returns: True if we need to early stop
        """
        if valid_loader is None:
            return False

        # Set model for evaluation
        self.eval()
        epoch_loss, epoch_score = 0, 0

        # We execute one inference step on validation set
        with no_grad():

            for item in valid_loader:

                # We extract the data
                x, y, _ = item

                # We perform the forward pass
                output = self(x)

                # We calculate the loss and the score
                epoch_loss += self.loss(output, y).item()
                epoch_score += self._eval_metric(output, y)

        # We update evaluations history
        mean_epoch_score = self._update_evaluations_progress(epoch_loss, epoch_score,
                                                             nb_batch=len(valid_loader),
                                                             mask_type=MaskType.VALID)

        # We check early stopping status
        early_stopper(mean_epoch_score, self)

        if early_stopper.early_stop:
            return True

        return False

    def forward(self, x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: (N, D') tensor with values of the node within the last layer

        """
        # We initialize a list of tensors to concatenate
        new_x = []

        # We extract continuous data
        if len(self._cont_idx) != 0:
            new_x.append(x[:, self._cont_idx])

        # We perform entity embeddings on categorical features not identified as genes
        if len(self._cat_idx) != 0:
            new_x.append(self._embedding_block(x))

        # We compute a genomic signature for categorical features identified as genes
        if self._genes_available:
            new_x.append(self._genes_dropout_layer(self._genes_encoding_block(x)))

        # We concatenate all inputs
        x = cat(new_x, 1)

        return self._linear_layer(self._main_encoding_block(x)).squeeze()


class MLPBinaryClassifier(MLP):
    """
    Multilayer perceptron binary classification model with entity embedding
    """
    def __init__(self,
                 layers: List[int],
                 activation: str,
                 eval_metric: Optional[Metric] = None,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 gene_idx_groups: Optional[Dict[str, List[int]]] = None,
                 gene_encoder_constructor: Optional[Callable] = None,
                 pos_weight: Optional[float] = None,
                 verbose: bool = False):
        """
        Sets the evaluation metric and then other protected attributes using parent's constructor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            gene_encoder_constructor: function that generates a GeneEncoder from gene_idx_groups
            pos_weight: scaling factor attributed to positive samples (samples in class 1)
            verbose: true to print training progress when fit is called
        """
        # We set the eval metric
        if eval_metric is None:
            eval_metric = BinaryCrossEntropy(pos_weight=pos_weight)
        else:
            if hasattr(eval_metric, 'pos_weight'):
                eval_metric.pos_weight = pos_weight

        super().__init__(output_size=1,
                         layers=layers,
                         activation=activation,
                         criterion=BCEWithLogitsLoss(pos_weight=pos_weight),
                         criterion_name='BCE',
                         eval_metric=eval_metric,
                         dropout=dropout,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         gene_idx_groups=gene_idx_groups,
                         gene_encoder_constructor=gene_encoder_constructor,
                         verbose=verbose)

    def predict_proba(self,
                      dataset: PetaleDataset,
                      mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the probabilities of being in class 1 for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: list of dataset idx for which we want to predict proba

        Returns: (N,) tensor
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a sigmoid
        with no_grad():
            return sigmoid(self(x))


class MLPRegressor(MLP):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self,
                 layers: List[int],
                 activation: str,
                 eval_metric: Optional[Metric] = None,
                 dropout: float = 0,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 gene_idx_groups: Optional[Dict[str, List[int]]] = None,
                 gene_encoder_constructor: Optional[Callable] = None,
                 verbose: bool = False):
        """
        Sets protected attributes using parent's constructor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            num_cont_col: number of numerical continuous columns
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            gene_encoder_constructor: function that generates a GeneEncoder from gene_idx_groups
            verbose: true to print training progress when fit is called
        """
        eval_metric = eval_metric if eval_metric is not None else RootMeanSquaredError()
        super().__init__(output_size=1,
                         layers=layers,
                         activation=activation,
                         criterion=MSELoss(),
                         criterion_name='MSE',
                         eval_metric=eval_metric,
                         dropout=dropout,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         gene_idx_groups=gene_idx_groups,
                         gene_encoder_constructor=gene_encoder_constructor,
                         verbose=verbose)

    def predict(self,
                dataset: PetaleDataset,
                mask: Optional[List[int]] = None) -> tensor:
        """
        Returns the predicted real-valued targets for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDatasets which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor or array with D-dimensional samples
                     - y : (N,) tensor or array with classification labels
                     - idx : (N,) tensor or array with idx of samples according to the whole dataset
            mask: list of dataset idx for which we want to make predictions

        Returns: (N,) tensor
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # Extraction of data
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass
        with no_grad():
            return self(x)
