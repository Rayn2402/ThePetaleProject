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
from src.models.blocks.genes_signature_block import GeneGraphAttentionEncoder, GeneGraphEncoder, GeneSignatureDecoder
from src.training.early_stopping import EarlyStopper
from src.training.sam import SAM
from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP
from src.utils.score_metrics import Direction
from torch import abs, eye, mean, mm, no_grad, pow, save, sum, tensor, zeros
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
                 gene_idx_groups: Dict[str, List[int]],
                 lr: float,
                 rho: float = 0.5,
                 batch_size: int = 25,
                 valid_batch_size: Optional[int] = None,
                 max_epochs: int = 200,
                 patience: int = 15,
                 hidden_size: int = 3,
                 signature_size: int = 10,
                 genes_emb_sharing: bool = False,
                 aggregation_method: str = 'att'):
        """
        Builds a GeneGraphEncoder that trains with self supervised learning

        Args:
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            lr: learning rate
            rho: if >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer,
                 otherwise, standard SGD optimizer with momentum will be used
            batch_size: size of the batches in the training loader
            valid_batch_size: size of the batches in the valid loader
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
                                                   'valid_batch_size': valid_batch_size,
                                                   'patience': patience,
                                                   'max_epochs': max_epochs})

        # We validate and save the fuzzyness parameter
        if aggregation_method not in PetaleGGE.AGGREGATION_METHODS:
            raise ValueError(f'Aggregation method must be in {PetaleGGE.AGGREGATION_METHODS}')

        # We validate the rho parameter
        if rho <= 0:
            raise ValueError('rho must be > 0')

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

        # We create the decoder
        self.__dec = GeneSignatureDecoder(chrom_composition_mat=self.__enc.build_chrom_composition_mat(),
                                          hidden_size=hidden_size,
                                          signature_size=signature_size)

        # We initialize the optimizer
        self.__optimizer = None

        # We initialize the Jaccard similarities tensor
        self.__jaccard = None

    @staticmethod
    def __create_dataloader(dataset: PetaleDataset,
                            mask: List[int],
                            batch_size: Optional[int] = None) -> DataLoader:
        """
        Creates a dataloader

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: list of idx to use for sampling
            batch_size: size of batches in the dataloader

        Returns: Dataloader
        """
        # We save the number of idx in the mask
        n = len(mask)

        # We adjust the batch size if needed
        batch_size = n if batch_size is None else min(n, batch_size)

        # If there is a single element in the last batch we drop it
        drop_last = (n % batch_size == 1)

        # We create the dataloader
        return DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(mask), drop_last=drop_last)

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

    def __execute_train_step(self, train_data: DataLoader) -> None:
        """
        Executes one training epoch

        Args:
            train_data: training dataloader

        Returns: None
        """

        # We put the model in training mode
        self.train()

        # We clear the gradients
        self.__optimizer.zero_grad()

        # We execute a training step for each mini batch
        for item in train_data:

            # We extract the data
            x, _, idx = item

            # We update the weights (the gradient is cleared within this function)
            self.__update_weights(x, idx)

    def __execute_valid_step(self, valid_data: DataLoader) -> float:
        """
        Executes an inference step on the validation data

        Args:
            valid_data: valid dataloader

        Returns: None
        """
        # We put the model in eval mode
        self.eval()

        # We initialize the epoch loss
        epoch_loss = 0

        # We execute an inference step on the valid set
        with no_grad():
            for item in valid_data:

                # We extract the data
                x, _, idx = item

                # We compute the loss
                epoch_loss += self.loss(self(x), idx).item()

        # We return the mean epoch loss
        return epoch_loss/len(valid_data)

    def __update_weights(self, x: tensor, idx: List[int]) -> None:
        """
        Executes a weights update using Sharpness-Aware Minimization (SAM) optimizer

        Note from https://github.com/davda54/sam :
            The running statistics are computed in both forward passes, but they should
            be computed only for the first one. A possible solution is to set BN momentum
            to zero to bypass the running statistics during the second pass.

        Args:
            x: (N, D) tensor with D-dimensional samples
            idx: list of idx associated to patients for which the encodings
                 were calculated

        Returns: None
        """

        # First forward-backward pass
        self.loss(self(x), idx).backward()
        self.__optimizer.first_step()

        # Second forward-backward pass
        self.__disable_running_stats()
        self.loss(self(x), idx).backward()
        self.__optimizer.second_step()

        # We enable running stats again
        self.__enable_running_stats()

    def __set_jaccard_similarities(self, dts: PetaleDataset) -> None:
        """
        Sets the jaccard similarities matrix using one hot encodings of genes

        The Jaccard similarity of two binary vectors is : M11/(n - M00)

        where M11 is the number of times that the two vectors share a 1 at
        the same index, M00 is the number of times that the two vectors share
        a 0 at the same index and n is the length of both vectors.

        Args:
            dts: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset

        Returns: None
        """
        # We get one hot encodings related to genes
        e = dts.get_genes_one_hot_encodings()

        # We calculate M11 for all patients pairs
        m11 = mm(e, e.t()) - eye(len(dts))

        # We calculate M00 for all patients pairs
        e = abs(e - 1)
        m00 = mm(e, e.t()) - eye(len(dts))

        # We save the Jaccard similarities
        self.__jaccard = (m11/(e.shape[1] - m00)).requires_grad_(False)

    def loss(self, x: tensor, idx: List[int]) -> tensor:
        """
        Computes the weighted squared differences between the signatures of patients
        and then add the mean squared differences between the genes' embeddings
        decoded from the signatures and the real embeddings.

        Finally, we divide the loss by 2.

        Args:
            x: (N, SIGNATURE_SIZE) tensor with genes' embeddings
            idx: list of idx associated to patients for which the encodings
                 were calculated

        Returns: (1,) tensor with the loss
        """
        # We calculate the squared differences between the encodings of all patients
        squared_diff = pow(x.unsqueeze(dim=1) - x, 2).sum(dim=2)

        # We take a subset of the jaccard similarities matrix and make
        # the coefficients sum to 1
        jaccard_sim = self.__jaccard[idx, :]
        jaccard_sim = jaccard_sim[:, idx]
        jaccard_sim /= jaccard_sim.sum()

        # We compute the loss associated to jaccard similarities
        jacc_loss = (jaccard_sim * squared_diff).sum()

        # We compute the loss associated to the decoding quality
        dec_loss = mean(sum(pow(self.__dec(x) - self.__enc.cache, 2), dim=(1, 2)))

        # We now calculate the lops
        return (jacc_loss + dec_loss)/2

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
        # Creation of the training dataloader
        train_dataloader = self.__create_dataloader(dataset, dataset.train_mask, self._train_params['batch_size'])

        # Creation of the valid dataloader
        valid_dataloader = self.__create_dataloader(dataset, dataset.valid_mask, self._train_params['valid_batch_size'])

        # Creation of the early stopper
        early_stopper = EarlyStopper(patience=self._train_params['patience'], direction=Direction.MINIMIZE)

        # Creation of the optimizer
        self.__optimizer = SAM(self.parameters(), Adam, rho=self._train_params['rho'], lr=self._train_params['lr'])

        # We set the jaccard similarities matrix
        self.__set_jaccard_similarities(dataset)

        # Self supervised train
        for epoch in range(self._train_params['max_epochs']):

            # We execute a training step
            self.__execute_train_step(train_data=train_dataloader)

            # We execute a valid step
            valid_loss = self.__execute_valid_step(valid_data=valid_dataloader)

            # Early stopping check
            early_stopper(valid_loss, self)

            if early_stopper.early_stop:
                print(f"\nEarly stopping occurred at epoch {epoch} with"
                      f" best_epoch = {epoch - self._train_params['patience']}"
                      f" and best self supervised training loss = {round(early_stopper.best_val_score, 4)}")
                break

    def forward(self, x: tensor) -> tensor:
        """
        Executes a forward pass with the encoder to create genomic signature

        Args:
            x: (N, D) tensor with D dimensional samples

        Returns: (N, NB_GENES, HIDDEN_SIZE) tensor with genes' embeddings
        """
        return self.__enc(x)

    def predict(self,
                dataset: PetaleDataset,
                mask: Optional[List[int]] = None) -> tensor:
        """
        Predicts the embeddings for idx of a given mask (default = test)

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: list of dataset idx for which we want to predict the embeddings

        Returns:
        """
        # We set the mask
        mask = mask if mask is not None else dataset.test_mask

        # We extract the appropriate set
        x, _, _ = dataset[mask]

        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a sigmoid
        with no_grad():
            return self(x)

    def save_model(self, path: str) -> None:
        """
        Saves the GeneGraphEncoder weights

        Args:
            path: save path

        Returns: None

        """
        save(self.__enc, os.path.join(path, "gge.pt"))

    @staticmethod
    def get_hps() -> List[HP]:
        return list(GGEHP())

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


class GGEHP:
    """
    GeneGraphEncoder's hyperparameters
    """
    BATCH_SIZE = NumericalIntHP("batch_size")
    LR = NumericalContinuousHP("lr")
    RHO = NumericalContinuousHP("rho")

    def __iter__(self):
        return iter([self.BATCH_SIZE, self.LR, self.RHO])