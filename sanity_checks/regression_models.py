"""
Author: Nicolas Raymond

This file is used to test the regression models implemented
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
    from src.data.processing.sampling import get_warmup_data, extract_masks
    from src.models.random_forest import PetaleRFR
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # Creation of the dataset
    df, target, cont_cols, _ = get_warmup_data(manager)
    warmup_numpy_dataset = PetaleDataset(df, target, cont_cols, classification=False)

    # Setting of the masks
    masks = extract_masks(join(Paths.MASKS, "l0_masks.json"), k=1, l=1)
    train_mask, valid_mask, test_mask = masks[0]['train'], masks[0]['valid'], masks[0]['test']
    warmup_numpy_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    # Extraction of data
    x_train_n, y_train_n = warmup_numpy_dataset[train_mask]
    x_valid_n, y_valid_n = warmup_numpy_dataset[valid_mask]
    x_test_n, y_test_n = warmup_numpy_dataset[test_mask]

    # Metrics
    metrics = [AbsoluteError(), Pearson(), RootMeanSquaredError()]

    """
    Training and evaluation of PetaleRFR
    """
    petale_rfr = PetaleRFR(n_estimators=500, max_samples=0.8)
    petale_rfr.fit(x_train_n, y_train_n)
    pred = petale_rfr.predict(x_test_n)
    print("Random Forest Regressor :")
    for m in metrics:
        print(f"\t{m.name} : {m(pred, tensor(y_test_n))}")
