"""
Filename: genes_signature_blocks.py

Author: Nicolas Raymond

Description: Defines the modules in charge of encoding
             and decoding the genomic signature associated to patients.

Date of last modification: 2022/02/08
"""

from src.models.abstract_models.encoder import Encoder
from src.models.blocks.mlp_blocks import BaseBlock, EntityEmbeddingBlock
from torch import bmm, einsum, exp, normal, tensor, zeros
from torch.nn import BatchNorm1d, Conv1d, Linear, Module, Parameter
from torch.nn.functional import leaky_relu, relu
from typing import Dict, List


class GeneEncoder(Encoder, Module):
    """
    Generates a signature (embedding) associated to an individual genes graph
    """
    def __init__(self,
                 gene_idx_groups: Dict[str, List[int]],
                 hidden_size: int = 3,
                 signature_size: int = 10,
                 genes_emb_sharing: bool = False):
        """
        Saves protected attributes and builds the entity embedding block

        Args:
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: final genomic signature size (output size)
            genes_emb_sharing: if True, genes will share the same entity embedding layer

        Returns: None
        """
        # We extract the genes idx
        self._gene_idx_groups = gene_idx_groups
        self._genes_idx = []
        for idx in self._gene_idx_groups.values():
            self._genes_idx.extend(idx)

        # We save the nb of genes, the hidden size and the nb of chromosomes
        self._hidden_size = hidden_size
        self._nb_genes = len(self._genes_idx)
        self._nb_chrom = len(self._gene_idx_groups.keys())

        # Setting of input and output sizes protected attributes
        Module.__init__(self)
        Encoder.__init__(self,
                         input_size=self._nb_genes,
                         output_size=signature_size)

        # Creation of entity embedding block
        self._entity_emb_block = EntityEmbeddingBlock(cat_sizes=[3]*self._nb_genes,
                                                      cat_emb_sizes=[self._hidden_size]*self._nb_genes,
                                                      cat_idx=self._genes_idx,
                                                      embedding_sharing=genes_emb_sharing)

        # Creation of a cache to store gene embeddings
        self._embedding_cache = None

    @property
    def cache(self) -> tensor:
        return self._embedding_cache

    @property
    def gene_idx_groups(self) -> Dict[str, List[int]]:
        return self._gene_idx_groups

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def build_chrom_composition_mat(self) -> tensor:
        """
        Builds a (NB_CHROM, NB_GENES) tensor where each element at the position
        i,j is a 1 if gene-j is part of chromosome-i and 0 otherwise

        Returns: (NB_CHROM, NB_GENES) tensor
        """
        # We save coordinates with values in the sparse matrix
        x_coords, y_coords = [], []
        y = 0
        for x, idx in enumerate(self._gene_idx_groups.values()):
            nb_genes_in_chrom = len(idx)
            x_coords += [x]*nb_genes_in_chrom
            y_coords += range(y, y + nb_genes_in_chrom)
            y += nb_genes_in_chrom

        # We set the sparse matrix
        mat = zeros(self._nb_chrom, self._nb_genes, requires_grad=False)
        mat[x_coords, y_coords] = 1

        return mat

    def _compute_genes_emb(self, x: tensor) -> tensor:
        """
        Extracts genes embeddings using the entity embedding block
        and saves them in a cache

        Args:
            x: (N, D) tensor with D-dimensional samples

        Returns: (N, NB_GENES, HIDDEN_SIZE) tensor
        """
        # Entity embedding on genes
        h = self._entity_emb_block(x)  # (N, D) -> (N, HIDDEN_SIZE*NB_GENES)

        # Resize embeddings
        h = h.reshape(h.shape[0], self._nb_genes, self._hidden_size)  # (N, NB_GENES, HIDDEN_SIZE)
        self._embedding_cache = h

        return h


class GeneGraphEncoder(GeneEncoder):
    """
    Generates a signature (embedding) associated to an individual genes graph
    with average of gene embeddings followed by a convolutional and a linear layer
    """
    def __init__(self,
                 gene_idx_groups: Dict[str, List[int]],
                 hidden_size: int = 3,
                 signature_size: int = 10,
                 genes_emb_sharing: bool = False):
        """
        Sets protected attributes with parent's constructor, the convolutional layer,
        the linear layer, the batch norm and sets other protected attributes using the Encoder constructor

        Args:
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: final genomic signature size (output size)
            genes_emb_sharing: if True, genes will share the same entity embedding layer
        """
        # Call of parent's constructor
        super().__init__(gene_idx_groups=gene_idx_groups,
                         hidden_size=hidden_size,
                         signature_size=signature_size,
                         genes_emb_sharing=genes_emb_sharing)

        # Creation of the matrix used to calculate the average of entity embeddings
        # within each chromosome. This matrix will not be updated
        self.__chrom_weight_mat = self.build_chrom_composition_mat()
        self.__chrom_weight_mat /= self.__chrom_weight_mat.sum(dim=1).reshape(-1, 1)

        # Convolutional layer that must be applied to each chromosome embedding
        self._conv_layer = Conv1d(in_channels=1,
                                  out_channels=1,
                                  kernel_size=(self._hidden_size,),
                                  stride=(self._hidden_size,))

        # Linear layer that gives the final signature
        self._linear_layer = Linear(self._nb_chrom, signature_size)

        # Batch norm layers
        self._bn1 = BatchNorm1d(self._nb_chrom)  # Normalizes after convolution layer
        self._bn2 = BatchNorm1d(signature_size)  # Normalizes final signatures

    @property
    def chrom_weight_mat(self) -> tensor:
        return self.__chrom_weight_mat

    def forward(self, x: tensor, fuzzyness: float = 0) -> tensor:
        """
        Executes the following actions on each element in the batch:

        - Applies entity embedding for each genes
        - Computes the averages of embedding for each chromosome
        - Concatenates the averages of each chromosome
        - Applies a 1D conv filter to the concatenated chromosome embeddings to have a single value per chromosome
        - Applies a linear layer to the results tensor of shape (N, NB_CHROM)
        - Applies batch norm

        Args:
            x: (N, D) tensor with D-dimensional samples
            fuzzyness: if fuzzyness > 0, noise sampled from a multivariate normal of
                       mean = 0 and sigma = fuzzyness will be added to gene embeddings

        Returns: (N, D') tensor where D' is the signature size
        """
        # Entity embedding on genes
        h = self._compute_genes_emb(x)  # (N, D) -> (N, NB_GENES, HIDDEN_SIZE)

        # Noise addition
        if fuzzyness > 0:
            h = h + normal(mean=zeros(h.shape), std=fuzzyness).requires_grad_(False)

        # Compute entity embedding averages per chromosome subgraphs for each individual
        # (NB_CHROM, NB_GENES)(N, NB_GENES, HIDDEN_SIZE) -> (N, NB_CHROM, HIDDEN_SIZE)
        h = einsum('ij,kjf->kif', self.__chrom_weight_mat, h)

        # Concatenate all chromosome averages side to side and apply relu
        # (N, NB_CHROM, HIDDEN_SIZE) -> (N, NB_CHROM*HIDDEN_SIZE)
        h = relu(h.reshape(h.shape[0], self._nb_chrom*self._hidden_size))

        # Add a dummy dimension
        h.unsqueeze_(dim=1)

        # Apply convolutional layer and RELU then squeeze for the linear layer
        h = relu(self._bn1(self._conv_layer(h).squeeze()))  # (N, 1, NB_CHROM*HIDDEN_SIZE) -> (N, NB_CHROM)

        # Apply linear layer and batch norm
        h = self._bn2(self._linear_layer(h))  # (N, NB_CHROM) -> (N, SIGNATURE_SIZE)

        return h


class GeneSignatureDecoder(Module):
    """
    From a signature given by the GeneGraphEncoder, this module tries to recover
    the original gene embeddings
    """
    def __init__(self,
                 chrom_composition_mat: tensor,
                 hidden_size: int,
                 signature_size: int = 10):

        """
        Builds the BaseBlock layer and saves the adjacency matrix related
        to the genome of patient

        Args:
            chrom_composition_mat: (NB_CHROM, NB_GENES) tensor where each element at the position
                                    i,j is a 1 if gene-j is part of chromosome-i and 0 otherwise
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: genomic signature size (input size)
        """

        # Call of parent's constructor
        super().__init__()

        # Saving of nb of chromosomes and number of genes
        self.__nb_genes = chrom_composition_mat.shape[1]
        self.__nb_chrom = chrom_composition_mat.shape[0]

        # Setting of matrix used to recover genes embedding for chromosome embedding
        self.__gene_weight_mat = Parameter(normal(mean=zeros(self.__nb_genes, self.__nb_chrom),
                                                  std=1.0).requires_grad_(True))
        self.__mask = chrom_composition_mat.clone().detach().bool().byte().t().requires_grad_(False)

        # Creation of BaseBlock (first layer of the decoder)
        self.__linear_layer = BaseBlock(input_size=signature_size,
                                        output_size=self.__nb_chrom,
                                        activation='ReLU')

        # Creation of convolutional layer
        self.__conv_layer = Conv1d(in_channels=1,
                                   out_channels=hidden_size,
                                   kernel_size=(1,),
                                   stride=(1,))

        # Batch norm layer to apply after convolution
        self._bn = BatchNorm1d(hidden_size)

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
        # Apply (linear layer -> batch norm -> activation) to signatures
        h = self.__linear_layer(x)  # (N, SIGNATURE_SIZE) -> (N, NB_CHROM)

        # Addition of a dummy dimension for convolutional layer
        h.unsqueeze_(dim=1)  # (N, NB_CHROM) -> (N, 1, NB_CHROM)

        # Apply convolutional layer, batch norm and relu
        h = relu(self._bn(self.__conv_layer(h)))  # (N, 1, NB_CHROM) -> (N, HIDDEN_SIZE, NB_CHROM)

        # Transposition of last dimensions
        h = h.transpose(1, 2)  # (N, HIDDEN_SIZE, NB_CHROM) -> (N, NB_CHROM, HIDDEN_SIZE)

        # Multiplication by gene_weight_mat to recover gene embeddings
        # the mask ensure that 0's are not updated in the gene_weight_mat
        # (NB_GENES, NB_CHROM)(N, NB_CHROM, HIDDEN_SIZE) -> (N, NB_GENES, HIDDEN_SIZE)
        h = einsum('ij,kjf->kif', (self.__gene_weight_mat*self.__mask), h)

        return h


class GeneGraphAttentionEncoder(GeneEncoder):
    """
    Generates a signature (embedding) associated to an individual genes graph
    using a gene attention layer and a chromosome attention layer
    """
    def __init__(self,
                 gene_idx_groups: Dict[str, List[int]],
                 hidden_size: int = 3,
                 signature_size: int = 10,
                 genes_emb_sharing: bool = False):
        """
        Sets protected attributes with parent's constructor, the gene attention layer,
        the chromosome attention layer, the linear layer and the batch norm

        Args:
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: final genomic signature size (output size)
            genes_emb_sharing: If True, genes will share the same entity embedding layer
        """
        # Call of parent's constructor
        super().__init__(gene_idx_groups=gene_idx_groups,
                         hidden_size=hidden_size,
                         signature_size=signature_size,
                         genes_emb_sharing=genes_emb_sharing)

        # Initialization of the gene attention layer
        self._gene_attention_layer = GeneAttentionLayer(chrom_composition_mat=self.build_chrom_composition_mat(),
                                                        hidden_size=self._hidden_size)

        # Initialization of the chromosome attention layer
        self._chrom_attention_layer = ChromAttentionLayer(hidden_size=self._hidden_size)

        # Initialization of the linear layer
        self._linear_layer = Linear(in_features=self._hidden_size,
                                    out_features=self._output_size)

        # Initialization of the batch norm
        self._bn = BatchNorm1d(num_features=self._output_size)

    def forward(self, x: tensor) -> tensor:
        """
        Executes the following actions on each element in the batch:

        - Applies entity embedding for each genes
        - Computes a weighted average of embeddings within each chromosome (using attention)
        - Computes a weighted average of chromosome embeddings (using attention)
        - Applies a linear layer to the results tensor of shape (N, NB_CHROM)
        - Applies batch norm

        Args:
            x: (N, D) tensor with D-dimensional samples

        Returns: (N, D') tensor where D' is the signature size
        """

        # Entity embedding on genes
        h = self._compute_genes_emb(x)  # (N, D) -> (N, NB_GENES, HIDDEN_SIZE)

        # Gene attention layer
        h = relu(self._gene_attention_layer(h))  # (N, NB_GENES, HIDDEN_SIZE) -> (N, NB_CHROM, HIDDEN_SIZE)

        # Chromosome attention layer
        h = relu(self._chrom_attention_layer(h))  # (N, NB_CHROM, HIDDEN_SIZE) -> (N, HIDDEN_SIZE)

        # Signature calculation
        return self._bn(self._linear_layer(h))


class GeneAttentionLayer(Module):
    """
    Module that calculates attention coefficient for each genes in each chromosomes
    graph and then use them to calculate chromosomes embedding.
    """
    def __init__(self,
                 chrom_composition_mat: tensor,
                 hidden_size: int):

        """
        Saves the chrom_composition_mat and initializes the attention matrix
        Args:
            chrom_composition_mat: (NB_CHROM, NB_GENES) tensor where each element at the position
                                   i,j is a 1 if gene-j is part of chromosome-i and 0 otherwise
            hidden_size: size of gene embeddings

        Returns: None
        """
        super().__init__()
        self.__chrom_composition_mat = chrom_composition_mat
        self.__attention = Parameter(normal(mean=zeros(chrom_composition_mat.shape[0], hidden_size),
                                            std=1.0).requires_grad_(True))

    def forward(self, x: tensor) -> tensor:
        """
        Does the following step for each element in the batch:
        - Calculates attention coefficients for each gene in each chromosome
        - Calculates chromosome embeddings using a weighted average of gene
          embeddings within each chromosome. The weights are the attention coefficients.

        Args:
            x: (N, NB_GENES, NB_GENES, HIDDEN_SIZE) tensor

        Returns: (N, NB_CHROM, HIDDEN_SIZE) tensor with chromosome embeddings
        """

        # We calculate a special Haddamard product to separate gene embeddings of each subgraph
        # (NB_CHROM, NB_GENES)(N, NB_GENES, HIDDEN_SIZE) -> (N, NB_CHROM, NB_GENES, HIDDEN_SIZE)
        h = einsum('ij,njk->nijk', self.__chrom_composition_mat, x)

        # We transpose last two dimensions
        h = h.transpose(2, 3)  # (N, NB_CHROM, NB_GENES, HIDDEN_SIZE) -> (N, NB_CHROM, HIDDEN_SIZE, NB_GENES)

        # We calculate attention coefficients for each gene within each chromosome
        # (NB_CHROM, HIDDEN_SIZE)(N, NB_CHROM, HIDDEN_SIZE, NB_GENES) -> (N, NB_CHROM, NB_GENES)
        att = einsum('ij,kijm->kim', self.__attention, h)
        mask = att.clone().detach().bool().byte()
        att = exp(leaky_relu(att))*mask
        att = att/att.sum(dim=2, keepdim=True)

        # We calculate the chromosome embeddings
        # (N, NB_CHROM, NB_GENES)(N, NB_GENES, HIDDEN_SIZE) -> (N, NB_CHROM, HIDDEN_SIZE)
        return bmm(att, x)


class ChromAttentionLayer(Module):
    """
    Module that calculates attention coefficient for each chromosome and
    then use them to calculate a weighted average of chromosome embeddings
    """
    def __init__(self, hidden_size: int):
        """
        Sets the attention vector

        Args:
            hidden_size: size of chromosome embeddings

        Returns: None
        """
        super().__init__()
        self.__attention = Parameter(normal(mean=zeros(1, hidden_size), std=1.0).requires_grad_(True))

    def forward(self, x: tensor) -> tensor:
        """
        Does the following step for each element in the batch:
        - Calculates attention coefficient for each chromosome
        - Calculates a weighted average of chromosome embeddings.
          The weights are the attention coefficients.

        Args:
            x: (N, NB_CHROM, HIDDEN_SIZE) tensor

        Returns: (N, HIDDEN_SIZE) tensor with weighted average chromosome embeddings
        """

        # We calculate the attention coefficients
        # (1, HIDDEN_SIZE)(N, HIDDEN_SIZE, NB_CHROM) -> (N, NB_CHROM)
        att = einsum('ij,njk->nk', self.__attention, x.transpose(1, 2))
        att = exp(leaky_relu(att))
        att = att/att.sum(dim=1, keepdim=True)

        # We calculate the weighted average of the chromosome embeddings
        # (N, 1, NB_CHROM)(N, NB_CHROM, HIDDEN_SIZE) -> (N, HIDDEN_SIZE)
        return bmm(att.unsqueeze(dim=1), x).squeeze()


