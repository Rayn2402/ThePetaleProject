"""
Filename: self_supervised_training.py

Author: Nicolas Raymond

Description: This file stores object built for self supervised training

Date of last modification: 2022/01/17
"""

from src.data.processing.datasets import PetaleDataset
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.models.blocks.genes_signature_block import GeneGraphEncoder, GeneSignatureDecoder
from src.training.early_stopping import EarlyStopper
from src.training.sam import SAM
from src.utils.score_metrics import Direction
from torch import mean, normal, sum, tensor, zeros
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Dict, List, Tuple


class SSGeneEncoderTrainer(Module):
    """
    Trains a gene graph encoder using self supervised training
    """
    def __init__(self,
                 gene_graph_encoder: GeneGraphEncoder,
                 fuzzyness: float = 0.01):
        """
        Saves the encoder, builds the adjacency matrix
        needed in the loss calculation and then creates a decoder

        Args:
            gene_graph_encoder: GeneGraphEncoder object
            fuzzyness: standard deviation of the multivariate normal distribution
                       from which noise will be sampled in order to add fuzzyness
                       to the signature coming out of the encoder
        """
        super().__init__()

        # We validate and save the fuzzyness parameter
        if fuzzyness < 0:
            raise ValueError('fuzzyness must be >= 0')
        self.__fuzzyness = fuzzyness

        # We save the encoder
        self.__enc = gene_graph_encoder

        # We create the decoder
        self.__dec = GeneSignatureDecoder(chrom_weight_mat=self.__enc.chrom_weight_mat,
                                          hidden_size=self.__enc.chrom_weight_mat,
                                          signature_size=self.__enc.output_size)

        # We initialize the optimizer
        self.__optimizer = None

    @property
    def encoder(self) -> GeneGraphEncoder:
        return self.__enc

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

    def fit(self,
            dataset: PetaleDataset,
            lr: float,
            rho: float = 0.5,
            batch_size: int = 25,
            max_epochs: int = 200,
            patience: int = 15) -> None:
        """
        Trains the encoder and the decoder using self supervised learning.
        Uses Sharpness-Aware Minimization by default.

        Args:
            dataset: PetaleDataset used to feed the training dataloader
            lr: learning rate
            rho: neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, standard SGD optimizer with momentum will be used
            batch_size: size of the batches in the training loader
            max_epochs: Maximum number of epochs for training
            patience: Number of consecutive epochs without training loss improvement allowed

        Returns: None
        """
        # Creation of the dataloader
        train_size = len(dataset.train_mask)
        batch_size = min(train_size, batch_size)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(dataset.train_mask),
                                drop_last=(train_size % batch_size == 1))
        nb_batch = len(dataloader)

        # Creation of the early stopper
        early_stopper = EarlyStopper(patience=patience,
                                     direction=Direction.MINIMIZE)

        # Creation of the optimizer
        self.__optimizer = SAM(self.parameters(), Adam, rho=rho, lr=lr)

        # Self supervised train
        for epoch in range(max_epochs):
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
                print(f"\nEarly stopping occurred at epoch {epoch} with best_epoch = {epoch - patience}"
                      f" and best self supervised training loss = {round(early_stopper.best_val_score, 4)}")
                break

    def forward(self, x: tensor) -> tensor:
        """
        Applies the encoder, add fuzzyness and then applies the decoder function

        Args:
            x: (N, D) tensor with D dimensional samples

        Returns: (N, NB_GENES, HIDDEN_SIZE) tensor with genes' embeddings
        """
        # Signature computation
        signature = self.__enc(x)

        # Return embeddings decoded from signature
        return self.__dec(signature + normal(mean=zeros(signature.shape, requires_grad=False), std=self.__fuzzyness))

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
                x_coords += [x]*(nb_genes_in_chrom - i - 1)
                y_coords += range(x + 1, nb_genes + nb_genes_in_chrom)
            nb_genes += nb_genes_in_chrom

        adj_mat = zeros(nb_genes, nb_genes, requires_grad=False)
        adj_mat[x_coords, y_coords] = 1
        adj_mat += adj_mat.t()

        return nb_genes, adj_mat






