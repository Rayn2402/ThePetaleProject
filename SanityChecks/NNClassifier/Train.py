from SQL.DataManager.Utils import PetaleDataManager
from Models.GeneralModels import NNClassifier
from Training.Training import NNTrainer
from Utils.visualization import visualize_epoch_losses
from Datasets.Sampling import LearningOneSampler
from torch import unique, manual_seed
from SQL.NewTablesScripts.constants import SEED
import numpy as np

if __name__ == '__main__':

    # OVERFIT TEST #

    """
    The training loss should move towards 0 while the validation
    loss should decrease at first and then increase
    """
    # We set the seed for the sampling part
    np.random.seed(SEED)

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")
    sampler = LearningOneSampler(dm=manager)

    # Loading of data
    all_data = sampler(k=1, l=1)
    train = all_data[0]["train"]
    valid = all_data[0]["valid"]

    cat, cont = train.X_cat.shape[1], train.X_cont.shape[1]
    cat_sizes = [len(unique(train.X_cat[:, i])) for i in range(cat)]

    print(f"\nOverfitting test...\n")

    # We set the seed for the model
    manual_seed(SEED)

    # Creation of a simple model
    Model = NNClassifier(num_cont_col=cont, output_size=3, layers=[10, 20, 20],
                         activation='ReLU', cat_sizes=cat_sizes)

    # Creation of a Trainer
    trainer = NNTrainer(Model, metric=None, lr=0.001, batch_size=20, weight_decay=0,
                        epochs=1000, early_stopping_activated=False)

    # Training for 1000 epochs
    t_loss, v_loss = trainer.fit(train, valid, verbose=False)

    # Visualization of the losses
    visualize_epoch_losses(t_loss, v_loss)

    # WEIGHT DECAY TEST #

    """
    The training loss should be higher since we are increasing the L2 Penalty
    """

    print(f"\nWeight decay test...\n")
    for decay in [0, 1, 2]:

        # We set the seed for the model
        manual_seed(SEED)

        # Creation of a simple model
        Model = NNClassifier(num_cont_col=cont, output_size=3, layers=[10, 20, 20],
                             activation='ReLU', cat_sizes=cat_sizes)

        # Creation of a Trainer
        trainer = NNTrainer(Model, metric=None, lr=0.001, batch_size=20, weight_decay=decay,
                            epochs=100, early_stopping_activated=False)

        # Training for 100 epochs
        t_loss, v_loss = trainer.fit(train, valid, verbose=True)
        print("\n")

        # Visualization of the losses
        visualize_epoch_losses(t_loss, v_loss)

    # EARLY STOPPING TEST #
    print(f"\nEarly stopping test...\n")
    # We set the seed for the model
    manual_seed(SEED)

    # Creation of a simple model
    Model = NNClassifier(num_cont_col=cont, output_size=3, layers=[10, 20, 20],
                         activation='ReLU', cat_sizes=cat_sizes)

    # Creation of a Trainer
    trainer = NNTrainer(Model, metric=None, lr=0.001, batch_size=20, weight_decay=0,
                        epochs=1000, early_stopping_activated=True)

    # Training for 1000 epochs
    t_loss, v_loss = trainer.fit(train, valid, verbose=True)

    # Visualization of the losses
    visualize_epoch_losses(t_loss, v_loss)

