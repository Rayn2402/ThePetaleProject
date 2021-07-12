"""
This file is used to validate the NNTrainer and RFTrainer
"""

import sys

from os.path import join, dirname, realpath
from sklearn.ensemble import RandomForestClassifier
from torch import tensor


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleRFDataset, PetaleNNDataset
    from src.data.processing.sampling import get_learning_one_data, extract_masks, SIGNIFICANT
    from src.models.nn_models import NNClassifier
    from src.training.training import NNTrainer, RFTrainer
    from src.utils.score_metrics import Accuracy
    from src.utils.visualization import visualize_class_distribution

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True, genes=SIGNIFICANT,
                                                    complications=[CARDIOMETABOLIC_COMPLICATIONS])

    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "l1_masks.json"), k=1, l=0)
    train_mask, valid_mask, test_mask = masks[0]["train"], masks[0]["valid"], masks[0]["test"]

    """
    NNTrainer test
    """

    # Creation of a dataset for neural nets
    nn_dataset = PetaleNNDataset(df, CARDIOMETABOLIC_COMPLICATIONS, cont_cols, cat_cols)
    nn_dataset.update_masks(train_mask=train_mask,
                            valid_mask=valid_mask,
                            test_mask=test_mask)

    # We look at the test mask composition
    test_mask_targets = nn_dataset.y[test_mask]
    visualize_class_distribution(test_mask_targets, {0: "no", 1: "yes"})

    # We look at the training mask composition
    train_mask_targets = nn_dataset.y[train_mask]
    visualize_class_distribution(train_mask_targets, {0: "no", 1: "yes"})

    # Creation a neural net for classification
    nb_cont_cols = len(cont_cols)
    cat_sizes = [len(v.items()) for v in nn_dataset.encodings.values()]
    model = NNClassifier(num_cont_col=nb_cont_cols, output_size=2, layers=[3],
                         activation='ReLU', cat_sizes=cat_sizes)

    # Initialization of a metric
    metric = Accuracy()

    # Creation of a trainer
    trainer = NNTrainer(model, metric, lr=0.005, batch_size=15, epochs=250, early_stopping=True, patience=100)

    # Training for 200 epochs
    t_loss, v_loss = trainer.fit(nn_dataset, visualization=True, verbose=True, path=dirname(__file__))

    # We check the accuracy on the test set
    test_x_cont, test_x_cat, test_y = nn_dataset[nn_dataset.test_mask]
    log_prob = trainer.predict(x_cont=test_x_cont, x_cat=test_x_cat, log_prob=True)
    acc = metric(log_prob, test_y)
    print(f"\nNN accuracy : {acc}")

    """
    RFTrainer test
    """

    # Creation of a dataset for random forest
    rf_dataset = PetaleRFDataset(df, CARDIOMETABOLIC_COMPLICATIONS, cont_cols, cat_cols)
    rf_dataset.update_masks(train_mask=train_mask,
                            valid_mask=valid_mask,
                            test_mask=test_mask)

    # Creation of a random forest
    rf_model = RandomForestClassifier()

    # Creation of a trainer
    rf_trainer = RFTrainer(rf_model, metric)

    # Training of the model
    rf_trainer.fit(rf_dataset)

    # We check the accuracy on the test set
    x, y = rf_dataset[rf_dataset.test_mask]
    log_prob = rf_trainer.predict(x=x)
    acc = metric(log_prob, tensor(y))
    print(f"\nRF accuracy : {acc}")


