"""
Author : Nicolas Raymond

This file is used to test classification models
"""

import sys
from os.path import join, dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from settings.paths import Paths
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset, PetaleStaticGNNDataset
    from src.data.processing.sampling import get_learning_one_data, extract_masks, SIGNIFICANT, ALL
    from src.data.processing.sampling import CARDIOMETABOLIC_COMPLICATIONS as CMC
    from src.data.processing.sampling import COMPLICATIONS
    from src.models.han_models import PetaleBinaryHANC
    from src.models.mlp_models import PetaleBinaryMLPC
    from src.models.random_forest import PetaleBinaryRFC
    from src.models.tabnet_models import PetaleTNC
    from src.models.xgboost_models import PetaleBinaryXGBC
    from src.utils.score_metrics import BinaryAccuracy, BinaryBalancedAccuracy, BinaryCrossEntropy,\
        BalancedAccuracyEntropyRatio

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # Creation of the dataset
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True, complications=[CMC],
                                                    genes=SIGNIFICANT)
    l1_numpy_dataset = PetaleDataset(df, CMC, cont_cols, cat_cols)
    l1_tensor_dataset = PetaleDataset(df, CMC, cont_cols, cat_cols, to_tensor=True)
    l1_gnn_dataset = PetaleStaticGNNDataset(df, CMC, cont_cols, cat_cols)

    # Setting of the masks
    masks = extract_masks(join(Paths.MASKS, "l1_masks.json"), k=1, l=1)
    train_mask, valid_mask, test_mask = masks[0]['train'], masks[0]['valid'], masks[0]['test']
    l1_numpy_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    l1_tensor_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    l1_gnn_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    cat_idx, cat_sizes = l1_numpy_dataset.cat_idx, l1_numpy_dataset.cat_sizes

    # Metrics
    metrics = [BinaryAccuracy(), BinaryBalancedAccuracy(), BinaryBalancedAccuracy("geometric_mean"),
               BinaryCrossEntropy(), BalancedAccuracyEntropyRatio()]

    # Extraction of data
    _, y_test_n, _ = l1_numpy_dataset[test_mask]
    _, y_test_t, _ = l1_tensor_dataset[test_mask]

    # Weights attributed to class 1
    weights = [0.5, 0.55]

    for w in weights:

        print(f"\nC1 weight : {w}")
        """
        Training and evaluation of PetaleRFR
        """
        petale_rfc = PetaleBinaryRFC(n_estimators=1000, max_samples=0.8, weight=w)
        petale_rfc.fit(l1_numpy_dataset)
        pred = petale_rfc.predict_proba(l1_numpy_dataset)
        print("Random Forest Classifier :")
        for m in metrics:
            print(f"\t{m.name} : {m(pred, y_test_n)}")

        """
        Training and evaluation of PetaleXGBC
        """
        petale_xgbc = PetaleBinaryXGBC(subsample=0.8, max_depth=6, weight=w)
        petale_xgbc.fit(l1_numpy_dataset)
        pred = petale_xgbc.predict_proba(l1_numpy_dataset)
        print("XGBoost Classifier :")
        for m in metrics:
            print(f"\t{m.name} : {m(pred, y_test_n)}")

        """
        Training and evaluation of PetaleTNC
        """
        petale_tnc = PetaleTNC(cat_idx=cat_idx, cat_sizes=cat_sizes, cat_emb_sizes=cat_sizes, device='cpu',
                               lr=0.08, max_epochs=300, patience=50, batch_size=35, n_steps=6,
                               n_d=4, n_a=4, gamma=1.5, weight=w)
        petale_tnc.fit(l1_numpy_dataset)
        pred = petale_tnc.predict_proba(l1_numpy_dataset)
        print("TabNet Classifier :")
        for m in metrics:
            print(f"\t{m.name} : {m(pred, y_test_n)}")

        """
        Training and evaluation of PetaleMLPC
        """
        petale_mlpc = PetaleBinaryMLPC(layers=[10, 10], activation="ReLU", alpha=0, beta=0, lr=0.05,
                                       batch_size=50, patience=250, max_epochs=1500,
                                       num_cont_col=len(cont_cols), weight=w)
        petale_mlpc.fit(l1_tensor_dataset)
        pred = petale_mlpc.predict_proba(l1_tensor_dataset)
        print("MLP Classifier :")
        for m in metrics:
            print(f"\t{m.name} : {m(pred, y_test_t)}")

        """
        Training and evaluation of PetaleBinaryHANC
        """
        petale_hanc = PetaleBinaryHANC(meta_paths=l1_gnn_dataset.get_metapaths(),
                                       in_size=len(l1_gnn_dataset.cont_cols), hidden_size=15, alpha=0, beta=0.001,
                                       num_heads=10, lr=0.01, batch_size=62, max_epochs=200, patience=100, weight=w,
                                       verbose=True)
        petale_hanc.fit(l1_gnn_dataset)
        pred = petale_hanc.predict_proba(l1_gnn_dataset)
        print("HAN Classifier :")
        for m in metrics:
            print(f"\t{m.name} : {m(pred, y_test_t)}")
