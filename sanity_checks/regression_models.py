"""
Author: Nicolas Raymond

This file is used to test the regression models implemented
"""

import sys
from os.path import join, dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from settings.paths import Paths
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.sampling import get_warmup_data, extract_masks
    from src.models.mlp import PetaleMLPR
    from src.models.random_forest import PetaleRFR
    from src.models.tabnet import PetaleTNR
    from src.models.xgboost_ import PetaleXGBR
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # Creation of the datasets
    df, target, cont_cols, _ = get_warmup_data(manager)
    warmup_numpy_dataset = PetaleDataset(df, target, cont_cols, classification=False)
    warmup_tensor_dataset = PetaleDataset(df, target, cont_cols, classification=False, to_tensor=True)

    # Setting of the masks
    masks = extract_masks(join(Paths.MASKS, "l0_masks.json"), k=1, l=1)
    train_mask, valid_mask, test_mask = masks[0]['train'], masks[0]['valid'], masks[0]['test']
    warmup_numpy_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    warmup_tensor_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    # Extraction of test data
    _, y_test_n, _ = warmup_numpy_dataset[test_mask]
    _, y_test_t, _ = warmup_tensor_dataset[test_mask]

    # Metrics
    metrics = [AbsoluteError(), Pearson(), RootMeanSquaredError()]

    """
    Training and evaluation of PetaleRFR
    """
    petale_rfr = PetaleRFR(n_estimators=500, max_samples=0.8)
    petale_rfr.fit(warmup_numpy_dataset)
    pred = petale_rfr.predict(warmup_numpy_dataset)
    print("Random Forest Regressor :")
    for m in metrics:
        print(f"\t{m.name} : {m(pred, y_test_n)}")

    """
    Training and evaluation of PetaleXGBR
    """
    petale_xgbr = PetaleXGBR(subsample=0.8, max_depth=8)
    petale_xgbr.fit(warmup_numpy_dataset)
    pred = petale_xgbr.predict(warmup_numpy_dataset)
    print("XGBoost Regressor :")
    for m in metrics:
        print(f"\t{m.name} : {m(pred, y_test_n)}")

    """
    Training and evaluation of PetaleTNR
    """
    petale_tnr = PetaleTNR(max_epochs=300, patience=50, batch_size=30,
                           lr=0.01, n_steps=5, n_d=8, n_a=8, gamma=1.5,)
    petale_tnr.fit(warmup_numpy_dataset)
    pred = petale_tnr.predict(warmup_numpy_dataset)
    print("TabNet Regressor :")
    for m in metrics:
        print(f"\t{m.name} : {m(pred, y_test_n)}")

    """
    Training and evaluation of PetaleMLPR
    """
    petale_mlpr = PetaleMLPR(layers=[], activation="ReLU", alpha=0, beta=0, lr=0.05,
                             patience=250, max_epochs=1500, batch_size=51, num_cont_col=len(cont_cols))
    petale_mlpr.fit(warmup_tensor_dataset)
    pred = petale_mlpr.predict(warmup_tensor_dataset)
    print("MLP Regressor :")
    for m in metrics:
        print(f"\t{m.name} : {m(pred, y_test_t)}")