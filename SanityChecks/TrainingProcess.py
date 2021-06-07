"""
This file is used to validate the NNTrainer and RFTrainer
"""


from Data.Datasets import PetaleNNDataset
from Data.Sampling import get_learning_one_data, extract_masks
from Models.GeneralModels import NNClassifier
from os.path import join, dirname
from SQL.constants import *
from SQL.DataManagement.Utils import PetaleDataManager
from Training.Trainer import NNTrainer
from Utils.score_metrics import SensitivityCrossEntropyRatio
from Utils.visualization import visualize_class_distribution


if __name__ == '__main__':

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True,
                                                    complications=[COMPLICATIONS])

    # Extraction of masks
    masks = extract_masks(join(dirname(dirname(__file__)), "Masks", "L1_masks.json"), k=1, l=0)

    # Creation of a dataset for neural nets
    nn_dataset = PetaleNNDataset(df, COMPLICATIONS, cont_cols, cat_cols)
    nn_dataset.update_masks(train_mask=masks[0]["train"],
                            valid_mask=masks[0]["valid"],
                            test_mask=masks[0]["test"])

    # Creation a neural net for classification
    nb_cont_cols = len(cont_cols)
    cat_sizes = [len(v.items()) for v in nn_dataset.encodings.values()]
    model = NNClassifier(num_cont_col=nb_cont_cols, output_size=2, layers=[10, 20, 10],
                         activation='ReLU', cat_sizes=cat_sizes)

    # Initialization of a metric
    metric = SensitivityCrossEntropyRatio()

    # Creation of a trainer
    trainer = NNTrainer(model, metric, lr=0.01, batch_size=30, weight_decay=0, epochs=50)

    # Training for 50 epochs
    t_loss, v_loss = trainer.fit(nn_dataset, visualization=True, path='')