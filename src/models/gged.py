"""
Filename: gged.py

Author: Nicolas Raymond

Description: This file defines the Petale Gene Graph Encoder (GGE) model used for self supervised learning.

Date of last modification: 2022/02/04
"""
import os

from src.data.processing.datasets import PetaleDataset
from src.models.abstract_models.base_models import PetaleEncoder
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.models.blocks.genes_signature_block import GeneGraphAttentionEncoder, GeneGraphEncoder
from src.training.early_stopping import EarlyStopper
from src.training.sam import SAM
from src.utils.score_metrics import Direction
from torch import mean, no_grad, pow, save, sum, tensor, zeros
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Dict, List, Optional, Tuple


class PetaleGGE(PetaleEncoder, Module):
    """
    Gene Graph Encoder model used to trained with self supervised learning
    """
    AVG = 'avg'
    ATT = 'att'
    AGGREGATION_METHODS = [AVG, ATT]

    def __init__(self,
                 gene_cols: List[str],
                 gene_idx_groups: Dict[str, List[int]],
                 lr: float,
                 rho: float = 0.5,
                 batch_size: int = 25,
                 max_epochs: int = 200,
                 patience: int = 15,
                 hidden_size: int = 3,
                 signature_size: int = 10,
                 genes_emb_sharing: bool = False,
                 aggregation_method: str = 'att'):
        """
        Builds a GeneGraphEncoder that trains with self supervised learning

        Args:
            gene_cols: list with names of columns related to genes
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            lr: learning rate
            rho: if >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, standard SGD optimizer with momentum will be used
            batch_size: size of the batches in the training loader
            max_epochs: Maximum number of epochs for training
            patience: Number of consecutive epochs without improvement
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: final genomic signature size (output size)
            genes_emb_sharing: if True, genes will share the same entity embedding layer
            aggregation_method: gene embeddings aggregation method
        """
        # We call parents' constructor
        Module.__init__(self)
        PetaleEncoder.__init__(self, train_params={'lr': lr,
                                                   'rho': rho,
                                                   'batch_size': batch_size,
                                                   'patience': patience,
                                                   'max_epochs': max_epochs})

        # We validate and save the fuzzyness parameter
        if aggregation_method not in PetaleGGE.AGGREGATION_METHODS:
            raise ValueError(f'Aggregation method must be in {PetaleGGE.AGGREGATION_METHODS}')

        # We validate the rho parameter
        if rho <= 0:
            raise ValueError('rho must be > 0')

        # We save the list of gene columns
        self.__gene_cols = gene_cols

        # We create the encoder
        if aggregation_method == PetaleGGE.AVG:
            self.__enc = GeneGraphEncoder(gene_idx_groups=gene_idx_groups,
                                          hidden_size=hidden_size,
                                          signature_size=signature_size,
                                          genes_emb_sharing=genes_emb_sharing)
        else:
            self.__enc = GeneGraphAttentionEncoder(gene_idx_groups=gene_idx_groups,
                                                   hidden_size=hidden_size,
                                                   signature_size=signature_size,
                                                   genes_emb_sharing=genes_emb_sharing)

        # We initialize the optimizer
        self.__optimizer = None

        # We initialize the one hot Jaccard similarities tensor
        self.__jaccard = None

    def __disable_running_stats(self) -> None:
        """
        Disables batch norm momentum when executing SAM optimization step

        Returns: None
        """
        self.apply(TorchCustomModel.disable_module_running_stats)

    def __enable_running_stats(self) -> None:
        """
        Restores batch norm momentum when executing SAM optimization step

        Returns: None
        """
        self.apply(TorchCustomModel.enable_module_running_stats)

    def __update_weights(self, x: tensor) -> float:
        """
        Executes a weights update using Sharpness-Aware Minimization (SAM) optimizer

        Note from https://github.com/davda54/sam :
            The running statistics are computed in both forward passes, but they should
            be computed only for the first one. A possible solution is to set BN momentum
            to zero to bypass the running statistics during the second pass.

        Args:
            x: (N, D) tensor with D-dimensional samples

        Returns: training loss
        """

        # First forward-backward pass
        loss = self.loss(self(x))
        loss.backward()
        self.__optimizer.first_step()

        # Second forward-backward pass
        self.__disable_running_stats()
        self.loss(self(x)).backward()
        self.__optimizer.second_step()

        # We enable running stats again
        self.__enable_running_stats()

        return loss.item()

    def loss(self, pred: tensor) -> tensor:
        """
        Computes the mean squared differences between the genes' embeddings
        decoded from the signatures and the real embeddings

        Args:
            pred: (N, NB_GENES, HIDDEN_SIZE) tensor with genes' embeddings

        Returns: (1,) tensor with the loss
        """
        return mean(sum(pow(pred - self.__enc.cache, 2), dim=(1, 2)))

    def fit(self, dataset: PetaleDataset) -> None:
        """
        Trains the encoder and the decoder using self supervised learning.
        Uses Sharpness-Aware Minimization by default.

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset


        Returns: None
        """
        # We put the model in training mode
        self.train()

        # Creation of the dataloader
        train_size = len(dataset.train_mask)
        batch_size = min(train_size, self._train_param['batch_size'])
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(dataset.train_mask),
                                drop_last=(train_size % batch_size == 1))
        nb_batch = len(dataloader)

        # Creation of the early stopper
        early_stopper = EarlyStopper(patience=self._train_param['patience'],
                                     direction=Direction.MINIMIZE)

        # Creation of the optimizer
        self.__optimizer = SAM(self.parameters(), Adam, rho=self._train_param['rho'], lr=self._train_param['lr'])

        # Self supervised train
        for epoch in range(self._train_param['max_epochs']):
            mean_epoch_loss = 0
            for batch in dataloader:

                # Data extraction
                x, _, _ = batch

                # Weight update
                mean_epoch_loss += self.__update_weights(x)

            # Mean epoch loss calculation
            mean_epoch_loss /= nb_batch

            # Early stopping check
            early_stopper(mean_epoch_loss, self)

            if early_stopper.early_stop:
                print(f"\nEarly stopping occurred at epoch {epoch} with"
                      f" best_epoch = {epoch - self._train_param['patience']}"
                      f" and best self supervised training loss = {round(early_stopper.best_val_score, 4)}")
                break

    def forward(self, x: tensor) -> tensor:
        """
        Execute a forward pass with the encoder to create genomic signature

        Args:
            x: (N, D) tensor with D dimensional samples

        Returns: (N, NB_GENES, HIDDEN_SIZE) tensor with genes' embeddings
        """
        return self.__enc(x)

    def predict(self,
                dataset: PetaleDataset,
                mask: Optional[List[int]] = None) -> tensor:
        """
        Predict the embeddings for idx of a given mask (default = train)

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: list of dataset idx for which we want to predict the embeddings

        Returns:
        """
        # We set the mask
        mask = mask if mask is not None else dataset.train_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a sigmoid
        with no_grad():
            return self.__enc(x)

    def save_model(self, path: str) -> None:
        """
        Saves the GeneGraphEncoder weights

        Args:
            path: save path

        Returns: None

        """
        save(self.__enc, os.path.join(path, "gge.pt"))

    @staticmethod
    def __set_adjacency_mat(gene_idx_groups: Dict[str, List[int]]) -> Tuple[int, tensor]:
        """
        Builds the adjacency matrix related to the genome graph identical
        to all patients

        Args:
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome

        Returns: nb of genes, adjacency matrix
        """
        nb_genes = 0
        x_coords, y_coords = [], []
        for idx in gene_idx_groups.values():
            nb_genes_in_chrom = len(idx)
            for i, x in enumerate(range(nb_genes, (nb_genes + nb_genes_in_chrom - 1))):
                x_coords += [x] * (nb_genes_in_chrom - i - 1)
                y_coords += range(x + 1, nb_genes + nb_genes_in_chrom)
            nb_genes += nb_genes_in_chrom

        adj_mat = zeros(nb_genes, nb_genes, requires_grad=False)
        adj_mat[x_coords, y_coords] = 1
        adj_mat += adj_mat.t()

        return nb_genes, adj_mat

