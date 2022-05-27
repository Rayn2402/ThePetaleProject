import matplotlib.pyplot as plt


from dgl import to_networkx
from matplotlib.cm import get_cmap, ScalarMappable
from networkx import connected_components, draw
from settings.paths import Paths
from src.data.extraction.constants import SEX
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.datasets import MaskType
from src.data.processing.gnn_datasets import PetaleKGNNDataset
from src.data.processing.sampling import get_warmup_data, extract_masks


if __name__ == '__main__':

    # Data extraction
    df, target, cont_col, cat_col = get_warmup_data(PetaleDataManager(), genes=None, sex=True, holdout=True)

    # Dataset construction
    dts = PetaleKGNNDataset(df=df, target=target, k=10,
                            cont_cols=cont_col, cat_cols=cat_col, self_loop=False,
                            conditional_cat_col=SEX, classification=False)

    # Train, valid, test idx extraction
    mask = extract_masks(Paths.WARMUP_HOLDOUT_MASK, k=1, l=0)
    train, valid, test = mask[0][MaskType.TRAIN], mask[0][MaskType.VALID], mask[0][MaskType.TEST]

    # Masks update
    dts.update_masks(train_mask=train, test_mask=test, valid_mask=valid)

    # Graph drawing
    plt.rc('text', usetex=True)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    g, idx_map, idx = dts.test_subgraph
    y = (dts.y - dts.y.min()) / (dts.y.max() - dts.y.min())
    c_map = get_cmap('viridis', len(g.nodes()))
    g = to_networkx(g).to_undirected()
    node_to_idx = {v: k for k, v in idx_map.items()}
    sex = ['Men', 'Women']
    for i, sg in enumerate([g.subgraph(c) for c in connected_components(g)]):
        original_idx = [node_to_idx[n] for n in sg.nodes()]
        color_map = c_map(y[original_idx])
        n_shape = 'D' if sex[i] == 'Men' else 'o'
        draw(sg, node_color=color_map, ax=axes[i], node_shape=n_shape)
        axes[i].collections[0].set_edgecolor('black')
        axes[i].set_title(sex[i])

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(cmap=c_map), cax=cbar_ax, ticks=[0, 1])
    cbar.set_label('VO$_2$ peak (ml/kg/min)')
    min, max = dts.y.min(), dts.y.max()
    cbar.ax.set_yticklabels([f'{min:.0f}', f'{max:.0f}'])
    for f in ['pdf', 'svg']:
        plt.savefig(f'graphs.{f}')
    plt.show()