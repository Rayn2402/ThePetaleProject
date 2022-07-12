"""
Filename: embeddings_projection.py

Author: Nicolas Raymond

Description: Script used to project patient embeddings in a 2D space
             using TSNE and compare a patient profile in the holdout set
             to the closest patients in the learning set

Date of last modification: 2022/07/12
"""
import sys

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from os.path import dirname, join, realpath
from sklearn.manifold import TSNE
from torch import load

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
    from hps.manually_selected_hps import GATHPS
    from settings.paths import Paths
    from src.data.extraction.constants import PARTICIPANT, SEX, TDM6_HR_END, TDM6_DIST
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import MaskType
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.sampling import extract_masks, get_VO2_data
    from src.models.gat import PetaleGATR

    # Data loading
    df, target, cont_col, cat_col = get_VO2_data(PetaleDataManager(), baselines=True, sex=True, holdout=True)
    df.drop([TDM6_HR_END, TDM6_DIST], axis=1, inplace=True)
    for f in [TDM6_HR_END, TDM6_DIST]:
        cont_col.remove(f)

    # Dataset construction
    dts = PetaleKGNNDataset(df, target, k=10, self_loop=True, classification=False,
                            cont_cols=cont_col, cat_cols=cat_col,
                            conditional_cat_col=SEX)

    # Mask extraction
    masks = extract_masks(Paths.VO2_HOLDOUT_MASK, k=1, l=0)
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
    gat_wrapper.model.load_state_dict(load(join(Paths.MODELS, 'vo2_gat.pt')))

    # Forward pass
    pred = gat_wrapper.predict(dts)

    # Embeddings extraction
    emb = gat_wrapper.model.emb_cache.squeeze()

    # Dataframe creation
    g, idx_map, idx = dts.test_subgraph

    """
    Embeddings projection figure with TSNE
    """
    # Subplots creation
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    men_idx, men_pos, women_idx, women_pos = [], [], [], []
    for i, id_ in enumerate(idx):
        if dts.original_data.iloc[id_][SEX] == 'Men':
            men_pos.append(i)
            men_idx.append(id_)
        elif dts.original_data.iloc[id_][SEX] == 'Women':
            women_pos.append(i)
            women_idx.append(id_)
        else:
            raise Exception('Missing sex')

    proj_x = TSNE(n_components=2, perplexity=10, random_state=1010710).fit_transform(emb.numpy())
    cmap = 'viridis'
    ax.scatter(proj_x[men_pos, 0], proj_x[men_pos, 1], c=dts.y[men_idx].numpy(), cmap=cmap, marker='D')
    ax.scatter(proj_x[women_pos, 0], proj_x[women_pos, 1], c=dts.y[women_idx].numpy(), cmap=cmap, marker='o')
    ax.set_title('Embeddings projection')

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(cmap=cmap), cax=cbar_ax, ticks=[0, 1])
    cbar.set_label('VO$_2$ peak (ml/kg/min)')
    min, max = dts.y.min(), dts.y.max()
    cbar.ax.set_yticklabels([f'{min:.0f}', f'{max:.0f}'])

    # Figure saving
    for f in ['pdf', 'svg']:
        plt.savefig(f'embeddings_projection.{f}')

    """
    Distance analysis
    """
    # Metric weights
    metric_weight = gat_wrapper.model.state_dict()['_linear_layer.weight'].abs()

    # Test embedding extraction
    test_emb = emb[idx_map[163]]

    # Training embeddings
    train_emb = emb[[idx_map[i] for i in dts.train_mask]]

    # Similarities calculation with weighted euclidean distance
    dist = ((test_emb - train_emb).pow(2)*metric_weight).sum(axis=1)
    similarities = 1/(1 + dist)
    dist_and_idx = sorted(zip(similarities.tolist(), dts.train_mask), key=lambda x: x[0], reverse=True)

    # Visualization of the test patient and his 3 closest neighbors
    patient_df = dts.original_data.iloc[163]
    print(patient_df[[PARTICIPANT] + cont_col + cat_col])
    print(f'Prediction: {pred[dts.test_mask.index(163)]:.2f}')

    for i in range(3):
        sim, index = dist_and_idx[i][0], dist_and_idx[i][1]
        patient_df = dts.original_data.iloc[index]
        print('\n\n')
        print(patient_df[[PARTICIPANT] + cont_col + cat_col])
        print(f'Similarity : {sim:.2f}, VO2 peak : {dts.y[index]:.2f}')





