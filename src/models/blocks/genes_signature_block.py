"""
Filename: genes_signature_blocks.py

Author: Nicolas Raymond

Description: Defines the modules in charge of encoding
             and decoding the genomic signature associated to patients.

Date of last modification: 2022/05/02
"""

from src.models.abstract_models.encoder import Encoder
from src.models.blocks.mlp_blocks import BaseBlock, EntityEmbeddingBlock
from torch import bmm, einsum, exp, max, normal, sigmoid, tensor, zeros
from torch.nn import AvgPool1d, BatchNorm1d, Conv1d, Dropout, Linear, Module, Parameter
from torch.nn.functional import leaky_relu, relu
from typing import Dict, List, Optional


class GeneEncoder(Encoder, Module):
    """
    Generates a signature (embedding) associated to an individual genes graph
    """
    def __init__(self,
                 gene_idx_groups: Dict[str, List[int]],
                 hidden_size: int = 2,
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
        self._gene_embedding_cache = None

    @property
    def gene_embedding_cache(self) -> tensor:
        return self._gene_embedding_cache

    @property
    def gene_idx_groups(self) -> Dict[str, List[int]]:
        return self._gene_idx_groups

    @property
    def genes_idx(self) -> List[int]:
        return self._genes_idx

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def nb_chrom(self) -> int:
        return self._nb_chrom

    @property
    def nb_genes(self) -> int:
        return self._nb_genes

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
        self._gene_embedding_cache = h

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
                 dropout: float = 0.5,
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
            dropout: dropout probability in the last hidden layer
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

        # Dropout layer
        self._dropout = Dropout(p=dropout)

    @property
    def chrom_weight_mat(self) -> tensor:
        return self.__chrom_weight_mat

    def forward(self, x: tensor) -> tensor:
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

        Returns: (N, D') tensor where D' is the signature size
        """
        # Entity embedding on genes
        h = self._compute_genes_emb(x)  # (N, D) -> (N, NB_GENES, HIDDEN_SIZE)

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

        # Apply dropout
        h = self._dropout(h)

        # Apply linear layer and batch norm
        h = self._bn2(self._linear_layer(h))  # (N, NB_CHROM) -> (N, SIGNATURE_SIZE)

        return h


class GeneSignatureDecoder(Module):
    """
    From a signature given by the GeneGraphEncoder, this module tries to recover
    the original gene embeddings
    """
    def __init__(self,
                 nb_genes: int,
                 signature_size: int = 10):

        """
        Builds the BaseBlock layer and saves the adjacency matrix related
        to the genome of patient

        Args:
            nb_genes: number of genes in patient genes data
            signature_size: genomic signature size (input size)
        """

        # Call of parent's constructor
        super().__init__()

        # Creation of BaseBlock (first layer of the decoder)
        self.__linear_layer = BaseBlock(input_size=signature_size,
                                        output_size=nb_genes,
                                        activation='ReLU')

        # Creation of first convolutional layer
        self.__conv_layer1 = Conv1d(in_channels=1,
                                    out_channels=nb_genes,
                                    kernel_size=(1,),
                                    stride=(1,))

        # Creation of second convolutional layer
        self.__conv_layer2 = Conv1d(in_channels=1,
                                    out_channels=3,
                                    kernel_size=(1,),
                                    stride=(1,))

        # Batch norm layers to apply after convolution
        self._bn1 = BatchNorm1d(nb_genes)
        self._bn2 = BatchNorm1d(3)

        # Average pooling layer
        self._avg_pooling = AvgPool1d(kernel_size=signature_size,
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
        # Addition of a dummy dimension for convolutional layer
        x.unsqueeze_(dim=1)  # (N, SIGNATURE_SIZE) -> (N, 1, SIGNATURE_SIZE)

        # Apply conv -> batch norm -> relu to signatures
        h = relu(self._bn1(self.__conv_layer1(x)))  # (N, SIGNATURE_SIZE) -> (N, NB_GENES, SIGNATURE_SIZE)

        # Apply mean pooling
        h = self._avg_pooling(h)  # (N, NB_GENES, SIGNATURE_SIZE) -> (N, NB_GENES, 1)

        # Transposition of last dimensions
        h = h.transpose(1, 2)  # (N, NB_GENES, 1) -> (N, 1, NB_GENES)

        # Apply convolutional layer, batch norm and sigmoid
        h = sigmoid(self._bn2(self.__conv_layer2(h)))  # (N, 1, NB_GENES) -> (N, 3, NB_GENES)

        # Transposition of last dimensions
        h = h.transpose(1, 2)  # (N, 3, NB_GENES) -> (N, NB_GENES, 3)

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
                 dropout: float = 0.5,
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
            dropout: dropout probability
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

        # Dropout layers
        self._dropout1 = Dropout(p=dropout)
        self._dropout2 = Dropout(p=dropout)

        # Attention cache
        self._att_dict = {}

    @property
    def att_dict(self) -> Dict[str, tensor]:
        return self._att_dict

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
        # (N, NB_GENES, HIDDEN_SIZE) -> (N, NB_CHROM, HIDDEN_SIZE)
        h = relu(self._gene_attention_layer(h, self.att_dict))

        # Second dropout layer
        h = self._dropout1(h)

        # Chromosome attention layer
        h = relu(self._chrom_attention_layer(h, self.att_dict))  # (N, NB_CHROM, HIDDEN_SIZE) -> (N, HIDDEN_SIZE)

        # Third dropout layer
        h = self._dropout2(h)

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

    def forward(self, x: tensor, att_dict: Optional[dict]) -> tensor:
        """
        Does the following step for each element in the batch:
        - Calculates attention coefficients for each gene in each chromosome
        - Calculates chromosome embeddings using a weighted average of gene
          embeddings within each chromosome. The weights are the attention coefficients.

        Args:
            x: (N, NB_GENES, HIDDEN_SIZE) tensor
            att_dict: dict in which attention scores will be stored

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
        att = exp(leaky_relu(att, negative_slope=0.2))*mask
        att = att/max(att.sum(dim=2, keepdim=True), tensor(1e-10).float())

        # We save the attention scores
        if att_dict is not None:
            att_dict['gene_att'] = att

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

    def forward(self, x: tensor, att_dict: Optional[dict]) -> tensor:
        """
        Does the following step for each element in the batch:
        - Calculates attention coefficient for each chromosome
        - Calculates a weighted average of chromosome embeddings.
          The weights are the attention coefficients.

        Args:
            x: (N, NB_CHROM, HIDDEN_SIZE) tensor
            att_dict: dict in which attention scores will be stored

        Returns: (N, HIDDEN_SIZE) tensor with weighted average chromosome embeddings
        """

        # We calculate the attention coefficients
        # (1, HIDDEN_SIZE)(N, HIDDEN_SIZE, NB_CHROM) -> (N, NB_CHROM)
        att = einsum('ij,njk->nk', self.__attention, x.transpose(1, 2))
        att = exp(leaky_relu(att, negative_slope=0.2))
        att = att/att.sum(dim=1, keepdim=True)

        # We save the attention scores
        if att_dict is not None:
            att_dict['chrom_att'] = att

        # We calculate the weighted average of the chromosome embeddings
        # (N, 1, NB_CHROM)(N, NB_CHROM, HIDDEN_SIZE) -> (N, HIDDEN_SIZE)
        return bmm(att.unsqueeze(dim=1), x).squeeze()


