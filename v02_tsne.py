"""
Filename: vo2_tsne.py

Author: Nicolas Raymond

Description: Script used to extract participants visualization

Date of last modification: 2022/05/16
"""

from hps.fixed_hps import GATHPS
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from os.path import join
from settings.paths import Paths
from sklearn.manifold import TSNE
from src.data.extraction.constants import PARTICIPANT, SEX, TDM6_HR_END, TDM6_DIST
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.datasets import MaskType
from src.data.processing.gnn_datasets import PetaleKGNNDataset
from src.data.processing.sampling import extract_masks, get_warmup_data
from src.models.gat import PetaleGATR
from torch import diag, load, mm


if __name__ == '__main__':

    # Data loading
    m = PetaleDataManager()
    df, target, cont_col, cat_col = get_warmup_data(m, baselines=True, sex=True, holdout=True)
    df.drop([TDM6_HR_END, TDM6_DIST], axis=1, inplace=True)
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
    pred = gat_wrapper.predict(dts)

    # Embeddings extraction
    emb = gat_wrapper.model.emb_cache.squeeze()

    # Dataframe creation
    g, idx_map, idx = dts.test_subgraph

    """
    TSNE figures
    """

    # TSNE Subplots creation
    plt.rc('text', usetex=True)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    x = TSNE(n_components=2, perplexity=10).fit_transform(emb.numpy())
    axes[0].scatter(x[:, 0], x[:, 1], c=dts.y[idx].numpy(), cmap='viridis')
    axes[0].set_title('GAT embeddings projection')

    x = TSNE(n_components=2, perplexity=10).fit_transform(dts.x.numpy())
    axes[1].scatter(x[:, 0], x[:, 1], c=dts.y.numpy(), cmap='viridis')
    axes[1].set_title('Original projection')

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(ScalarMappable(cmap='viridis'), cax=cbar_ax)

    # Figure saving
    for f in ['pdf', 'svg']:
        plt.savefig(f'embeddings_projection.{f}')
    plt.show()
    plt.close()

    """
    Distance analysis
    """

    # Metric weights
    print(gat_wrapper.model.state_dict())
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





