"""
Filename: genes_signature_blocks.py

Author: Nicolas Raymond

Description: Defines the modules in charge of encoding
             and decoding the genomic signature associated to patients.

Date of last modification: 2022/01/17
"""

from src.models.abstract_models.encoder import Encoder
from src.models.blocks.mlp_blocks import BaseBlock, EntityEmbeddingBlock
from torch import einsum, tensor, zeros
from torch.nn import BatchNorm1d, Conv1d, Linear, Module
from torch.nn.functional import relu
from typing import Dict, List


class GeneGraphEncoder(Encoder, Module):
    """
    Generates a signature (embedding) associated to an individual genes graph
    """
    def __init__(self,
                 gene_idx_groups: Dict[str, List[int]],
                 hidden_size: int = 3,
                 signature_size: int = 10):
        """
        Builds the entity embedding block, the convolutional layer, the linear layer,
        the batch norm and sets other protected attributes using the Encoder constructor

        Args:
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: final genomic signature size (output size)
        """
        # We extract the genes idx
        self.__gene_idx_groups = gene_idx_groups
        self.__genes_idx = []
        for idx in self.__gene_idx_groups.values():
            self.__genes_idx.extend(idx)

        # We save the nb of genes, the hidden size and the nb of chromosomes
        self.__hidden_size = hidden_size
        self.__nb_genes = len(self.__genes_idx)
        self.__nb_chrom = len(self.__gene_idx_groups.keys())

        # Setting of input and output sizes protected attributes
        Module.__init__(self)
        Encoder.__init__(self,
                         input_size=self.__nb_genes,
                         output_size=signature_size)

        # Creation of entity embedding block
        self._entity_emb_block = EntityEmbeddingBlock(cat_sizes=[3]*self.__nb_genes,
                                                      cat_emb_sizes=[self.__hidden_size]*self.__nb_genes,
                                                      cat_idx=self.__genes_idx)

        # Creation of the matrix used to calculate the average of entity embeddings
        # within each chromosome. This matrix will not be updated
        self.__chrom_weight_mat = zeros(self.__nb_chrom, self.__nb_genes, requires_grad=False)
        self.__set_chromosome_weight_mat(self.__gene_idx_groups)

        # Convolutional layer that must be applied to each chromosome embedding
        self._conv_layer = Conv1d(in_channels=1,
                                  out_channels=1,
                                  kernel_size=(self.__hidden_size,),
                                  stride=(self.__hidden_size,))

        # Linear layer that gives the final signature
        self._linear_layer = Linear(self.__nb_chrom, signature_size)

        # Batch norm layer that normalize final signatures
        self._bn = BatchNorm1d(signature_size)

        # Creation of a cache needed to calculate loss
        self.__embedding_cache = None

    @property
    def cache(self) -> tensor:
        return self.__embedding_cache

    @property
    def chrom_weight_mat(self) -> tensor:
        return self.__chrom_weight_mat

    @property
    def gene_idx_groups(self) -> Dict[str, List[int]]:
        return self.__gene_idx_groups

    @property
    def hidden_size(self) -> int:
        return self.__hidden_size

    def __set_chromosome_weight_mat(self, gene_idx_groups: Dict[str, List[int]]) -> None:
        """
        Sets the matrix used to calculate mean of entity embeddings within each chromosome

        Args:
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome

        Returns: None
        """
        # We save coordinates with values in the sparse matrix
        x_coords, y_coords = [], []
        y = 0
        for x, idx in enumerate(gene_idx_groups.values()):
            nb_genes_in_chrom = len(idx)
            x_coords += [x]*nb_genes_in_chrom
            y_coords += range(y, y + nb_genes_in_chrom)
            y += nb_genes_in_chrom

        # We set the sparse matrix with weights to calculate averages
        # for each chromosome graph
        self.__chrom_weight_mat[x_coords, y_coords] = 1
        self.__chrom_weight_mat /= self.__chrom_weight_mat.sum(dim=1).reshape(-1, 1)

    def forward(self, x: tensor) -> tensor:
        """
        Executes the following actions on each element in the batch:

        - Applies entity embedding for each genes
        - Computes the averages of embedding for each chromosome
        - Concatenates the averages of each chromosome
        - Applies a 1D conv filter to the concatenated chromosome embeddings to have a single value per chromosome
        - Applies a linear layer to the results tensor of shape (N, NB_CHROM)

        Args:
            x: (N, D) tensor with D-dimensional samples

        Returns: (N, D') tensor where D' is the signature size
        """
        # Entity embedding on genes
        h = self._entity_emb_block(x)  # (N, D) -> (N, HIDDEN_SIZE*NB_GENES)

        # Resize embeddings
        self.__embedding_cache = h.reshape(h.shape[0], self.__nb_genes, self.__hidden_size)  # (N, NB_GENES, HIDDEN_SIZE)

        # Compute entity embedding averages per chromosome subgraphs for each individual
        # (NB_CHROM, NB_GENES)x(N, NB_GENES, HIDDEN_SIZE) -> (N, NB_CHROM, HIDDEN_SIZE)
        h = einsum('ij,kjf->kif', self.__chrom_weight_mat, self.__embedding_cache)

        # Concatenate all chromosome averages side to side
        # (N, NB_CHROM, HIDDEN_SIZE) -> (N, NB_CHROM*HIDDEN_SIZE)
        h = h.reshape(h.shape[0], self.__nb_chrom*self.__hidden_size)

        # Add a dummy dimension
        h.unsqueeze_(dim=1)

        # Apply convolutional layer and RELU then squeeze for the linear layer
        h = relu(self._conv_layer(h)).squeeze()  # (N, 1, NB_CHROM*HIDDEN_SIZE) -> (N, NB_CHROM)

        # Apply linear layer and batch norm
        h = self._bn(self._linear_layer(h))  # (N, NB_CHROM) -> (N, SIGNATURE_SIZE)

        return h


class GeneSignatureDecoder(Module):
    """
    From a signature given by the GeneGraphEncoder, this module tries to recover
    the original gene embeddings
    """
    def __init__(self,
                 chrom_weight_mat: tensor,
                 hidden_size: int,
                 signature_size: int = 10):

        """
        Builds the BaseBlock layer and saves the adjacency matrix related
        to the genome of patient

        Args:
            chrom_weight_mat: (NB_CHROM, NB_GENES) matrix used to calculate average of embeddings within a chromosome
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: genomic signature size (input size)
        """

        # Call of parent's constructor
        super().__init__()

        # Saving of nb of chromosomes and number of genes
        self.__nb_genes = chrom_weight_mat.shape[1]
        self.__nb_chrom = chrom_weight_mat.shape[0]

        # Setting of matrix used to recover genes embedding for chromosome embedding
        self.__gene_weight_mat = self.__build_gene_weight_mat(chrom_weight_mat=chrom_weight_mat)

        # Creation of BaseBlock (first layer of the decoder)
        self.__linear_layer = BaseBlock(input_size=signature_size,
                                        output_size=self.__nb_chrom,
                                        activation='ReLU')

        # Creation of convolutional layer
        self.__conv_layer = Conv1d(in_channels=1,
                                   out_channels=hidden_size,
                                   kernel_size=(1,),
                                   stride=(1,))

    def forward(self, x: tensor) -> tensor:
        """
        Executes the following actions on each element in the batch:

        - Applies linear layer -> relu -> batch norm to have a single value per chromosome
        - Applies a 1D conv filter to each chromosome value in order to get "hidden size" feature map
        - Applies the inverse operation of the average that was done in the encoder
          to recover gene embeddings.

        Args:
            x: (N, D') tensor with D'-dimensional samples

        Returns: (N, NB_GENES, HIDDEN_SIZE) tensor with genes' embeddings
        """
        # Apply (linear layer -> activation -> batch norm) to signatures
        h = self.__linear_layer(x)  # (N, SIGNATURE_SIZE) -> (N, NB_CHROM)

        # Addition of a dummy dimension for convolutional layer
        h.unsqueeze_(dim=1)  # (N, NB_CHROM) -> (N, 1, NB_CHROM)

        # Apply convolutional layer
        h = self.__conv_layer(h)  # (N, 1, NB_CHROM) -> (N, NB_CHROM, HIDDEN_SIZE)

        # Multiplication by gene_weight_mat to recover gene embeddings
        # (NB_GENES, NB_CHROM)x(N, NB_CHROM, HIDDEN_SIZE) -> (N, NB_GENES, HIDDEN_SIZE)
        h = einsum('ij,kjf->kif', self.__gene_weight_mat, h)

        return h

    @staticmethod
    def __build_gene_weight_mat(chrom_weight_mat: tensor) -> tensor:
        """
        Builds the matrix used to recover genes' embeddings from the chromosomes' embeddings

        Args:
            chrom_weight_mat: (NB_CHROM, NB_GENES) tensor used to calculate average of embeddings within a chromosome

        Returns: (NB_GENES, NB_CHROM) tensor

        """
        gene_mat = (chrom_weight_mat.clone().detach()).t()
        gene_mat /= pow(gene_mat.sum(dim=1).reshape(-1, 1), 2)

        return gene_mat
