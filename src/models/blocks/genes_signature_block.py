"""
Filename: genes_signature_blocks.py

Author: Nicolas Raymond

Description: Defines the modules in charge of encoding
             and decoding the genomic signature associated to patients.

Date of last modification: 2022/01/04
"""

from src.models.abstract_models.encoder import Encoder
from src.models.blocks.mlp_blocks import EntityEmbeddingBlock
from torch import tensor, zeros
from torch.nn import Module
from typing import Dict, List


class GeneGraphEncoder(Encoder, Module):
    """
    Generates a signature (embedding) associated to an individual genes graph
    """
    def __init__(self,
                 genes_idx_group: Dict[str, List[int]],
                 hidden_size: int = 3,
                 signature_size: int = 10):
        """
        Builds the entity embedding block and sets other protected attributes
        using the Encoder constructor

        Args:
            genes_idx_group: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: final genomic signature size (output size)
        """
        # We extract the genes idx
        self.__genes_idx = []
        for idx in genes_idx_group.values():
            self.__genes_idx.extend(idx)

        # We save the nb of genes and nb of chromosomes
        self.__nb_genes = len(self.__genes_idx)
        self.__nb_chrom = len(genes_idx_group.keys())

        # Setting of input and output sizes protected attributes
        Module.__init__(self)
        Encoder.__init__(self, input_size=self.__nb_genes, output_size=signature_size)

        # Creation of entity embedding block
        self._entity_emb_block = EntityEmbeddingBlock(cat_sizes=[3]*self.__nb_genes,
                                                      cat_emb_sizes=[hidden_size]*self.__nb_genes,
                                                      cat_idx=self.__genes_idx)

        # Creation of the matrix used to calculate mean of entity embeddings
        # within each chromosome. This matrix will not be updated
        self.__chrom_weight_mat = zeros(self.__nb_chrom, self.__nb_genes, requires_grad=False)
        self.__set_chromosome_weight_mat(genes_idx_group)

    def __set_chromosome_weight_mat(self, genes_idx_group: Dict[str, List[int]]) -> None:
        """
        Sets the matrix used to calculate mean of entity embeddings within each chromosome

        Args:
            genes_idx_group: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome

        Returns: None
        """
        # We save coordinates with values in the sparse matrix
        x_coords, y_coords = [], []
        y = 0
        for x, idx in enumerate(genes_idx_group.values()):
            nb_genes_in_chrom = len(idx)
            x_coords += [x]*nb_genes_in_chrom
            y_coords += range(y, y + nb_genes_in_chrom)
            y += nb_genes_in_chrom

        # We set the sparse matrix with weights the calculate means
        # for each chromosome graph
        self.__chrom_weight_mat[x_coords, y_coords] = 1
        self.__chrom_weight_mat /= self.__chrom_weight_mat.sum(dim=1).reshape(-1, 1)

    def forward(self, x: tensor) -> tensor:
        """
        Executes the following actions on each element in the batch:

        - Applies entity embedding for each genes in the graph
        - Computes the means of embedding for each chromosome
        - Concatenates the means of each chromosome
        - Applies a 1D convolution filter to the concatenated chromosome embeddings
        - Applies a linear layer to the results tensor of shape (N, nb_chromosome)

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: (N, D') tensor where D' is the signature size
        """

