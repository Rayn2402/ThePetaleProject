"""
Filename: vo2_attention_heat_map.py

Author: Nicolas Raymond

Description: Script used to extract participants attention scores on the holdout set

Date of last modification: 2022/05/05
"""

from hps.fixed_hps import GATHPS
from matplotlib import pyplot as plt
from os.path import join
from pandas import DataFrame, merge
from seaborn import heatmap
from settings.paths import Paths
from src.data.extraction.constants import PARTICIPANT, SEX, TDM6_HR_END, TDM6_DIST
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.datasets import MaskType
from src.data.processing.gnn_datasets import PetaleKGNNDataset
from src.data.processing.sampling import extract_masks, get_warmup_data
from src.models.gat import PetaleGATR
from torch import load


if __name__ == '__main__':

    # Data loading
    m = PetaleDataManager()
    df, target, cont_col, cat_col = get_warmup_data(m, baselines=True, sex=True, holdout=True)
    for f in [TDM6_HR_END, TDM6_DIST]:
        cont_col.remove(f)

    # Dataset construction
    dts = PetaleKGNNDataset(df, target, k=10, self_loop=True, classification=False,
                            cont_cols=cont_col, cat_cols=cat_col,
                            conditional_cat_col=SEX)

    # Mask extraction
    masks = extract_masks(Paths.WARMUP_HOLDOUT_MASK, k=1, l=0)
    test_mask = masks[0][MaskType.TEST]

    # Masks update
    dts.update_masks(train_mask=masks[0][MaskType.TRAIN],
                     valid_mask=masks[0][MaskType.VALID],
                     test_mask=test_mask)

    # Model construction
    gat_wrapper = PetaleGATR(num_cont_col=len(dts.cont_idx),
                             cat_idx=dts.cat_idx,
                             cat_sizes=dts.cat_sizes,
                             cat_emb_sizes=dts.cat_sizes,
                             max_epochs=500,
                             patience=50,
                             **GATHPS)

    # Model parameters loading
    gat_wrapper.model.load_state_dict(load(join(Paths.MODELS, 'warmup_gat.pt')))

    # Forward pass
    y = gat_wrapper.predict(dts)

    # Attention extraction
    att = gat_wrapper.model.att_cache.squeeze()

    # Dataframe creation
    g, idx_map, _ = dts.test_subgraph
    u, v = g.edges()[0].tolist(), g.edges()[1].tolist()
    idx_to_ids = {v: dts.ids[k] for k, v in idx_map.items()}
    u = [idx_to_ids[node] for node in u]
    v = [idx_to_ids[node] for node in v]

    data = {}
    for i, id_ in enumerate(v):
        if id_ not in data:
            data[id_] = {}
        data[id_].update({u[i]: att[i].item()})

    # Heatmap data setting
    heat_map_df = DataFrame.from_dict(data, orient='index')
    heat_map_df.index.name = PARTICIPANT
    sex_df = df.set_index(PARTICIPANT)[[SEX]]
    heat_map_df = merge(heat_map_df, sex_df, right_index=True, left_index=True)
    heat_map_df.sort_values(by=[SEX, PARTICIPANT], inplace=True)
    heat_map_df = heat_map_df[heat_map_df.index]
    heat_map_df[heat_map_df.isna()] = 0

    # Heatmap creation
    heatmap(heat_map_df)
    plt.ylabel(None)
    plt.tight_layout()
    for f in ['pdf', 'svg']:
        plt.savefig(f'warmup_genes_att_heatmap.{f}')




