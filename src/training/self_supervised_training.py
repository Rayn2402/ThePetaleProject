"""
Filename: self_supervised_training.py

Author: Nicolas Raymond

Description: This file stores object built for self supervised training

Date of last modification: 2022/01/12
"""

from src.models.blocks.genes_signature_block import GeneGraphEncoder, GeneSignatureDecoder
from torch import Module, tensor, zeros
from typing import Dict, List, Tuple


class SSGeneEncoderTrainer(Module):
    """
    Trains a gene graph encoder using self supervised training
    """
    def __init__(self, gene_graph_encoder: GeneGraphEncoder):
        """
        Saves the encoder, builds the adjacency matrix
        needed in the loss calculation and then creates a decoder

        Args:
            gene_graph_encoder: GeneGraphEncoder object
        """

        # We save the encoder
        self.__enc = gene_graph_encoder

        # We count of the number of genes and create the adjacency matrix
        nb_genes, self.__adj_mat = self.__set_adjacency_mat(self.__enc.gene_idx_groups)

        # We create the decoder
        self.__dec = GeneSignatureDecoder(nb_genes=nb_genes,
                                          signature_size=self.__enc.output_size)

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




