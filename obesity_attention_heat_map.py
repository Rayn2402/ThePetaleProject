"""
Filename: obesity_attention_heat_map.py

Author: Nicolas Raymond

Description: Script used to extract genes attention on the holdout set

Date of last modification: 2022/05/02
"""

from hps.fixed_hps import ENET_GGE_HPS
from matplotlib import pyplot as plt
from os.path import join
from pandas import DataFrame
from seaborn import heatmap
from settings.paths import Paths
from src.data.extraction.constants import PARTICIPANT
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.datasets import MaskType, PetaleDataset
from src.data.processing.sampling import extract_masks, GeneChoice, get_learning_one_data
from src.models.blocks.genes_signature_block import GeneEncoder, GeneGraphAttentionEncoder
from src.models.mlp import PetaleMLPR
from torch import einsum, load
from typing import Dict, List, Optional


# Selected genes
SELECTED_GENES = ['6_110760008', '17_26096597', '17_37884037', '17_4856580',
                  '1_66036441', '21_44324365', '17_38545824', '13_23824818',
                  '1_161182208', '6_29912280', '12_48272895', '6_29912386',
                  '22_42486723', '7_45932669', '7_87160618', '21_37518706',
                  '1_226019633', '16_88713236', '15_58838010', '6_29912333',
                  '7_20762646', '13_95863008', '2_46611678', '17_48712711',
                  '7_94946084', '2_240946766', '4_120241902', '6_12296255',
                  '16_69745145']

SELECTED_FEATURES = ['34500 Sex', 'Methotrexate', 'DOX (mg/m2)',
                     'Effective corticosteroid cumulative dose (mg/m2)',  '34472 Age at diagnosis',
                     'Duration of treatment']

if __name__ == '__main__':

    # Data loading
    m = PetaleDataManager()
    df, target, cont_col, cat_col = get_learning_one_data(m, GeneChoice.ALL, baselines=True, holdout=True)
    cat_col = [c for c in cat_col if c in SELECTED_FEATURES] + SELECTED_GENES
    cont_col = [c for c in cont_col if c in SELECTED_FEATURES]
    dts = PetaleDataset(df, target, cont_col, cat_col, gene_cols=SELECTED_GENES, to_tensor=True)

    # Mask extraction
    masks = extract_masks(Paths.OBESITY_HOLDOUT_MASK, k=1, l=0)
    test_mask = masks[0][MaskType.TEST]
    dts.update_masks(train_mask=masks[0][MaskType.TRAIN],
                     valid_mask=masks[0][MaskType.VALID],
                     test_mask=test_mask)

    # Model creation
    def gene_encoder_constructor(gene_idx_groups: Optional[Dict[str, List[int]]],
                                 dropout: float) -> GeneEncoder:
        """
        Builds a GeneGraphAttentionEncoder

        Args:
            gene_idx_groups: dictionary where keys are names of chromosomes and values
                             are list of idx referring to columns of genes associated to
                             the chromosome
            dropout: dropout probability

        Returns: GeneEncoder
        """

        return GeneGraphAttentionEncoder(gene_idx_groups=gene_idx_groups,
                                         genes_emb_sharing=False,
                                         dropout=dropout,
                                         signature_size=4)

    ggae_wrapper = PetaleMLPR(max_epochs=500,
                              patience=50,
                              num_cont_col=len(dts.cont_idx),
                              cat_idx=dts.cat_idx,
                              cat_sizes=dts.cat_sizes,
                              cat_emb_sizes=dts.cat_sizes,
                              gene_idx_groups=dts.gene_idx_groups,
                              gene_encoder_constructor=gene_encoder_constructor,
                              **ENET_GGE_HPS)

    # Model parameters loading
    ggae_wrapper.model.load_state_dict(load(join(Paths.MODELS, 'obesity_ggae.pt')))

    # Forward pass
    y = ggae_wrapper.predict(dts)
    att_dict = ggae_wrapper.model.att_dict

    # Attention extraction
    chrom_att, in_chrom_gene_att = att_dict['chrom_att'], att_dict['gene_att']
    gene_att = einsum('ijk,ijn->ijn', chrom_att.unsqueeze(dim=2), in_chrom_gene_att) # (N, NB_CHROM, 1)(N, NB_CHROM, NB_GENES)
    gene_att = gene_att.sum(axis=1)

    # Dataframe creation
    participants = [dts.ids[i] for i in test_mask]
    idx_to_gene = {v: k for k, v in dts._gene_idx.items()}
    genes = []
    for v in dts.gene_idx_groups.values():
        for g in v:
            genes.append(idx_to_gene[g])

    data = {}
    for i, p in enumerate(participants):
        data[p] = {gene: gene_att[i, j].item() for j, gene in enumerate(genes)}

    heat_map_df = DataFrame.from_dict(data, orient='index')

    # Heatmap creation
    genes_df = dts.get_imputed_dataframe(include_ids_column=True, include_target_column=True).set_index(PARTICIPANT)
    genes_df = genes_df.iloc[test_mask]
    genes_df.sort_values(target, inplace=True)

    heat_map_df = heat_map_df.join(genes_df[[target]])
    heat_map_df.sort_values(target, inplace=True)
    heat_map_df.drop([target], axis=1, inplace=True)
    heat_map_df = heat_map_df[sorted(heat_map_df.columns, key=lambda x: int(x.split('_')[0]))]

    genes_df = genes_df[heat_map_df.columns]
    genes_df.replace("0/0", 0, inplace=True)
    genes_df.replace("0/1", 1, inplace=True)
    genes_df.replace("1/1", 2, inplace=True)
    genes_df = genes_df.astype(int)

    heatmap(heat_map_df, annot=genes_df, annot_kws={"fontsize": 8}, xticklabels=True)
    plt.tight_layout()
    for f in ['pdf', 'svg']:
        plt.savefig(f'obesity_genes_att_heatmap.{f}')





