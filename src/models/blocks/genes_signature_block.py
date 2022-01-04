"""
Filename: genes_signature_blocks.py

Author: Nicolas Raymond

Description: Defines the modules in charge of encoding
             and decoding the genomic signature associated to patients.

Date of last modification: 2022/01/04
"""

from src.models.abstract_models.encoder import Encoder
from src.models.blocks.mlp_blocks import EntityEmbeddingBlock
from torch.nn import Module
from typing import List


class GeneGraphEncoder(Encoder, Module):
    """
    Generates a signature (embedding) associated to an individual genes graph
    """
    def __init__(self,
                 genes_idx: List[int],
                 genes_sizes: List[int],
                 hidden_size: int = 3,
                 signature_size: int = 10):
        """
        Builds the entity embedding block and sets other protected attributes
        using the Encoder constructor

        Args:
            genes_idx: idx of categorical columns related to genes
            genes_sizes: number of different categories for each gene
            hidden_size: embedding size of each genes during intermediate
                         signature creation procedure
            signature_size: final genomic signature size (output size)
        """
        # Setting of input and output sizes protected attributes
        nb_genes = len(genes_idx)
        Module.__init__(self)
        Encoder.__init__(self, input_size=nb_genes, output_size=signature_size)

        # Creation of entity embedding block
        self._entity_emb_block = EntityEmbeddingBlock(cat_sizes=genes_sizes,
                                                      cat_emb_sizes=[hidden_size]*nb_genes,
                                                      cat_idx=genes_idx)


