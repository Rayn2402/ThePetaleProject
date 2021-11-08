"""
Filename: meta_learner_experiment.py

Authors: Nicolas Raymond

Description: This file is used to store the meta learner experiment.
             We first train a linear regression, make a prediction on all the data points and
             register scores on the test set.

             We then train an heterogeneous graph attention network using only the predictions
             of the linear prediction as features.

Date of last modification : 2021/11/08
"""

import sys
from numpy import array
from os.path import dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset, PetaleStaticGNNDataset
    from src.data.processing.sampling import extract_masks, GeneChoice, get_warmup_data, MaskType
    from src.models.mlp import PetaleMLPR
    from src.models.han import PetaleHANR
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError, SquaredError

    # Initialization of DataManager and sampler
    manager = PetaleDataManager()

    # Creation of the first datasets
    df, target, cont_cols, cat_cols = get_warmup_data(manager, sex=True)
    lin_dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False,
                                to_tensor=True)

    # Extraction of the mask
    masks = extract_masks(Paths.WARMUP_MASK, k=1, l=1)
    train_mask, valid_mask, test_mask = masks[0][MaskType.TRAIN], masks[0][MaskType.VALID], masks[0][MaskType.TEST]
    lin_dataset.update_masks(train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)

    # Extraction of test data
    _, y, _ = lin_dataset[test_mask]

    # Metrics
    metrics = [AbsoluteError(), Pearson(), SquaredError(), RootMeanSquaredError()]

    """
    Training and evaluation of linear regression
    """
    linear_reg = PetaleMLPR(n_layer=0, n_unit=5, activation="ReLU", alpha=1, beta=1, lr=0.06,
                            patience=25, max_epochs=300, batch_size=35, num_cont_col=len(cont_cols))
    linear_reg.fit(lin_dataset)
    pred = linear_reg.predict(lin_dataset, mask=list(range(len(lin_dataset))))
    print("Linear regression:")
    for m in metrics:
        print(f"\t{m.name} : {m(pred[test_mask], y)}")

    """
    Training and evaluation of meta learner (HAN)
    """
    # Creation of new dataset
    df, target, cont_cols, cat_cols = get_warmup_data(manager, genes=GeneChoice.ALL)
    df.drop(cont_cols, inplace=True, axis=1)
    cont_cols = ['feature_1']
    df[cont_cols[0]] = array(pred)
    han_dataset = PetaleStaticGNNDataset(df, target, cont_cols, cat_cols, classification=False)
    han_dataset.update_masks(train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)

    # Training
    han_reg = PetaleHANR(meta_paths=han_dataset.get_metapaths(), in_size=1,
                         hidden_size=1, num_heads=20, lr=0.05, batch_size=25, max_epochs=100,
                         patience=25, alpha=1, beta=1, verbose=True)
    han_reg.fit(han_dataset)
    pred = han_reg.predict(han_dataset, mask=list(range(len(han_dataset))))
    for m in metrics:
        print(f"\t{m.name} : {m(pred[test_mask], y)}")
