"""
Author : Nicolas Raymond

This file is used to test classification models
"""

import sys
from os.path import join, dirname, realpath
from torch import tensor

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from settings.paths import Paths
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.sampling import get_learning_one_data, extract_masks
    from src.data.processing.sampling import CARDIOMETABOLIC_COMPLICATIONS as CMC
    from src.models.random_forest import PetaleBinaryRFC
    from src.models.tabnet_models import PetaleTNC
    from src.models.xgboost_models import PetaleBinaryXGBC
    from src.utils.score_metrics import BinaryAccuracy, BinaryBalancedAccuracy, BinaryCrossEntropy,\
        BalancedAccuracyEntropyRatio

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # Creation of the dataset
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True, complications=[CMC])
    l1_numpy_dataset = PetaleDataset(df, CMC, cont_cols, cat_cols, classification=False)

    # Setting of the masks
    masks = extract_masks(join(Paths.MASKS, "l1_masks.json"), k=1, l=1)
    train_mask, valid_mask, test_mask = masks[0]['train'], masks[0]['valid'], masks[0]['test']
    l1_numpy_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    cat_idx, cat_sizes = l1_numpy_dataset.cat_idx, l1_numpy_dataset.cat_sizes

    # Metrics
    metrics = [BinaryAccuracy(), BinaryBalancedAccuracy(), BinaryBalancedAccuracy("geometric_mean"),
               BinaryCrossEntropy(), BalancedAccuracyEntropyRatio()]

    # Extraction of data
    x_train_n, y_train_n = l1_numpy_dataset[train_mask]
    x_valid_n, y_valid_n = l1_numpy_dataset[valid_mask]
    x_test_n, y_test_n = l1_numpy_dataset[test_mask]

    # Weights attributed to class 1
    weights = [0.5, 0.55, 0.65, 0.75, 1]

    for w in weights:

        print(f"\nC1 weight : {w}")
        """
        Training and evaluation of PetaleRFR
        """
        petale_rfc = PetaleBinaryRFC(n_estimators=1000, max_samples=0.8, weight=w)
        petale_rfc.fit(x_train_n, y_train_n)
        pred = petale_rfc.predict_proba(x_test_n)
        print("Random Forest Classifier :")
        for m in metrics:
            print(f"\t{m.name} : {m(pred, y_test_n)}")

        """
        Training and evaluation of PetaleXGBC
        """
        petale_xgbc = PetaleBinaryXGBC(subsample=0.8, max_depth=6, weight=w)
        petale_xgbc.fit(x_train_n, y_train_n)
        pred = petale_xgbc.predict_proba(x_test_n)
        print("XGBoost Classifier :")
        for m in metrics:
            print(f"\t{m.name} : {m(pred, y_test_n)}")

        """
        Training and evaluation of PetaleTNC
        """
        petale_tnc = PetaleTNC(cat_idxs=cat_idx, cat_dims=cat_sizes, cat_emb_dim=cat_sizes, device='cpu',
                               lr=0.08, n_steps=6, n_d=4, n_a=4, gamma=1.5, weight=w)
        petale_tnc.fit(x_train_n, y_train_n, eval_set=[(x_valid_n, y_valid_n)],
                       max_epochs=300, patience=50, batch_size=35)
        pred = petale_tnc.predict_proba(x_test_n)
        print("TabNet Classifier :")
        for m in metrics:
            print(f"\t{m.name} : {m(pred, y_test_n)}")