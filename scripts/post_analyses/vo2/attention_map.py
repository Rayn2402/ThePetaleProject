"""
Filename: attention_map.py

Author: Nicolas Raymond

Description: Script used to extract attention between patient 190 and other
             patients from the training set.

Date of last modification: -
"""

import sys
from json import load as jsload
from os.path import dirname, join, realpath
from matplotlib import pyplot as plt
from matplotlib import ticker
from numpy import isclose
from pandas import DataFrame
from seaborn import heatmap
from torch import load, topk
from typing import Dict, List, Optional


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
    from settings.paths import Paths
    from src.data.extraction.constants import PARTICIPANT, SEX, TDM6_DIST, TDM6_HR_END
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import MaskType, PetaleDataset
    from src.data.processing.sampling import extract_masks, get_VO2_data
    from src.models.gas import PetaleGASR
    from src.recording.recording import Recorder

    # 1. Set the number of the data split for which P111 is in the test set
    SPLIT: int = 2

    # 2. Load the data
    df, target, cont_cols, cat_cols = get_VO2_data()

    # 3. Remove the sex
    df.drop([SEX], axis=1, inplace=True)
    cat_cols = None

    # 4. Create the dataset
    dts = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True)

    # 5. Add the predictions of the past model as a variable

    # 5.0 Load the predictions
    pred_path = join(Paths.EXPERIMENTS_RECORDS, 'experiment_with_walk', 'original_equation')
    with open(join(pred_path, f"Split_{SPLIT}", Recorder.RECORDS_FILE), "r") as read_file:
        data = jsload(read_file)

    # 5.1 Create the conversion function to extract predictions from strings
    def convert(x: str) -> List[float]:
        return [float(x)]

    # 5.2 Extract the predictions
    pred = {}
    for section in [Recorder.TRAIN_RESULTS, Recorder.TEST_RESULTS, Recorder.VALID_RESULTS]:
        if section in data.keys():
            pred = {**pred, **{p_id: [p_id, *convert(v[Recorder.PREDICTION])] for p_id, v in data[section].items()}}

    # 5.3 Creation a pandas dataframe
    df = DataFrame.from_dict(pred, orient='index', columns=[PARTICIPANT, 'pred0'])

    # 6. Create the new augmented dataset
    dts = dts.create_superset(data=df, categorical=False)

    # 7. Extract and set the mask
    masks = extract_masks(Paths.VO2_MASK, k=3, l=0)
    test_mask = masks[2][MaskType.TEST]
    dts.update_masks(train_mask=masks[2][MaskType.TRAIN],
                     valid_mask=masks[2][MaskType.VALID],
                     test_mask=test_mask)

    # 8. Locate patient P111 in the dataset
    row_idx = dts.ids_to_row_idx['P111']

    # 9. Locate the patient in the test mask
    mask_pos = test_mask.index(row_idx)

    # 10. Create the model
    epn_wrapper = PetaleGASR(previous_pred_idx=len(dts.cont_idx) - 1,
                             pred_mu=dts.original_data['pred0'].mean(),
                             pred_std=dts.original_data['pred0'].std(),
                             num_cont_col=len(dts.cont_idx),
                             cat_idx=dts.cat_idx,
                             cat_sizes=dts.cat_sizes,
                             cat_emb_sizes=dts.cat_sizes)

    # 11. Load the parameters of the model
    path = join(Paths.EXPERIMENTS_RECORDS, 'experiment_with_walk',
                'GASOE_vo2_automated_ns', 'Split_2', 'torch_model.pt')

    epn_wrapper.model.load_state_dict(load(path))

    # 12. Execute the forward pass and load the attention scores
    y = epn_wrapper.predict(dts)
    attn = epn_wrapper.model.attn_cache

    # 13. Extract the row associated to P111
    attn = attn[mask_pos]

    # 14. Identify the 10 patients with the highest attention scores
    _, pos_idx = topk(attn, k=10)

    # 15. Identify their original position in the dataset
    batch_idx = dts.train_mask + dts.test_mask
    idx = [batch_idx[i] for i in pos_idx]




    #
    # # Dataframe creation
    # participants = [dts.ids[i] for i in test_mask]
    # idx_to_gene = {v: k for k, v in dts._gene_idx.items()}
    # genes = []
    # for v in dts.gene_idx_groups.values():
    #     for g in v:
    #         genes.append(idx_to_gene[g])
    #
    # data = {}
    # for i, p in enumerate(participants):
    #     data[p] = {gene: gene_att[i, j].item() for j, gene in enumerate(genes)}
    #
    # heat_map_df = DataFrame.from_dict(data, orient='index')
    #
    # # Heatmap creation
    # plt.rc('text', usetex=True)
    # genes_df = dts.get_imputed_dataframe(include_ids_column=True, include_target_column=True).set_index(PARTICIPANT)
    # genes_df = genes_df.iloc[test_mask]
    # genes_df.sort_values(target, inplace=True)
    #
    # heat_map_df = heat_map_df.join(genes_df[[target]])
    # heat_map_df.sort_values(target, inplace=True)
    # heat_map_df.drop([target], axis=1, inplace=True)
    # heat_map_df = heat_map_df[sorted(heat_map_df.columns, key=lambda x: int(x.split('_')[0]))]
    #
    # genes_df = genes_df[heat_map_df.columns]
    # genes_df.replace("0/0", 0, inplace=True)
    # genes_df.replace("0/1", 1, inplace=True)
    # genes_df.replace("1/1", 2, inplace=True)
    # genes_df = genes_df.astype(int)
    #
    # heat_map_df.rename(columns={c: c.replace("_", ":") for c in heat_map_df.columns}, inplace=True)
    # genes_df.rename(columns={c: c.replace("_", ":") for c in genes_df.columns}, inplace=True)
    #
    # fig, ((ax0, dummy_ax), (ax1, cbar_ax)) = plt.subplots(nrows=2, ncols=2, sharex='col',
    #                                                       gridspec_kw={'height_ratios': [1, 10], 'width_ratios': [20, 1]})
    # heatmap(heat_map_df, annot=genes_df, annot_kws={"fontsize": 8}, cbar_ax=cbar_ax, xticklabels=True, ax=ax1)
    # cbar_ax.set_ylabel('Attention')
    # ax1.set_ylabel('Survivors in the holdout set')
    # ax1.set_xlabel('SNPs')
    #
    # # Histogram creation
    # att_means = heat_map_df.mean().to_numpy()
    # ax0.bar([i + 0.5 for i in range(len(att_means))], att_means, width=0.8, color='grey')
    # func = lambda x, pos: "" if isclose(x, 0) else x
    # ax0.spines.right.set_visible(False)
    # ax0.spines.top.set_visible(False)
    # ax0.xaxis.set_visible(False)
    # ax0.yaxis.set_major_formatter(ticker.FuncFormatter(func))
    # dummy_ax.axis('off')
    #
    # plt.tight_layout()
    # for f in ['pdf', 'svg']:
    #     plt.savefig(f'obesity_snps_att_heatmap.{f}')





