"""
Authors : Mehdi Mitiche
          Nicolas Raymond

Files that contains class related to the Trainer of the models

"""
import ray
import torch

from abc import ABC, abstractmethod
from src.data.processing.datasets import PetaleNNDataset, PetaleRFDataset
from numpy import mean, std, array, log
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from torch import optim, manual_seed, cuda, tensor
from torch import device as device_
from torch.nn import Module
from torch.utils.data import DataLoader, SubsetRandomSampler
from src.training.early_stopping import EarlyStopping
from typing import Optional, Callable, Tuple, Any, Union, Dict, List
from src.utils.score_metrics import Metric
from src.utils.visualization import visualize_epoch_progression


class Trainer(ABC):
    """
    Abstract class for the object that will train and evaluate a given model
    """
    def __init__(self, model: Optional[Any], metric: Optional[Metric], device: str = "cpu"):
        """
        Sets protected attributes

        Args:
            model: model to train
            metric: callable metric we want to optimize (not used for backpropagation)
            device: device used to run our training ("cpu" or "gpu")
        """
        # We call super init since we're using ABC
        super().__init__()

        # We save protected attributes
        self._model = model
        self._device_type = device
        self._device = device_("cuda:0" if cuda.is_available() and device == "gpu" else "cpu")
        self._metric = metric
        self._subprocess_defined = False
        self._subprocess = None

    @property
    def metric(self) -> Metric:
        return self._metric

    def inner_random_subsampling(self, n_splits: int, seed: Optional[int] = None) -> float:
        """
        Performs multiple random subsamplings validation on the model

        Args:
            n_splits: number of random subsampling splits
            seed: starting point to generate random numbers

        Returns: mean of scores divided by their standard deviation

        """
        # We make sure that a subprocess has been defined
        assert self._subprocess_defined, "The parallelizable subprocess must be defined before use"

        # We set the seed value
        if seed is not None:
            manual_seed(seed)

        # We train and test on each of the inner split
        futures = [self._subprocess.remote(i) for i in range(n_splits)]
        scores = ray.get(futures)

        # We the mean of the scores divided by the standard deviation
        standard_dev = 1 if len(scores) == 1 else std(scores)
        return mean(scores) / standard_dev

    def define_subprocess(self, dataset: Union[PetaleNNDataset, PetaleRFDataset],
                          masks: Dict[int, Dict[str, List[int]]]) -> None:
        """
        Builds the subprocess function according to the masks and the device

        Args:
            dataset: custom dataset on which we apply masks
            masks: dict with list of indexes

        """

        # We inform the trainer that a subprocess has been defined
        self._subprocess_defined = True

        # We build the subprocess according to the datasets
        gpus = 0.10 if (self._device_type != "cpu") else 0

        @ray.remote(num_gpus=gpus)
        def subprocess(i: int) -> float:
            """
            Consists of the parallelizable process of the inner random subsampling loop

            :param i: Index of the random subsamples' splits on which to test hyperparameters selection
            """
            # We the get the train, valid and test masks
            train_mask, valid_mask, test_mask = masks[i]["train"], masks[i]["valid"], masks[i]["test"]

            # We update dataset's masks
            dataset.update_masks(train_mask, valid_mask, test_mask)

            # We train our model with the train and valid sets
            self.fit(dataset=dataset)

            # We extract x_cont, x_cat and targets from the test set
            inputs, targets = self.extract_data(dataset[test_mask])

            # We get the predictions
            predictions = self.predict(**inputs, log_prob=True)

            # We flatten the predictions array if we are doing regression
            if predictions.shape[1] == 1:
                predictions = predictions.flatten()

            # We calculate the score with the help of the metric function
            score = self._metric(predictions, targets)

            # We save the scores
            return score

        # We set the subprocess internal method
        self._subprocess = subprocess

    @abstractmethod
    def extract_data(self, data: Tuple[Optional[Union[DataFrame, tensor]],
                                       Optional[Union[DataFrame, tensor]], Union[array, tensor]]
                     ) -> Tuple[Dict[str, Optional[Union[DataFrame, tensor]]], Union[array, tensor]]:
        """
        Abstract method that extracts data from a sliced dataset

        Args:
            data: sliced dataset/mini batch

        Returns: continuous data, categorical data and targets

        """

        raise NotImplementedError

    @abstractmethod
    def fit(self, dataset: Union[PetaleNNDataset, PetaleRFDataset], **kwargs) -> Optional[Tuple[tensor, tensor]]:
        """
        Abstract method to train and evaluate the model

        Args:
            dataset: custom dataset with masks already defined

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs) -> Union[tensor, array]:
        """
        Abstract method that returns predictions of a model
        (log probabilities in case of classification and real-valued numbers in case of regression)
        """
        raise NotImplementedError

    @abstractmethod
    def update_trainer(self, **kwargs) -> None:
        """
        Abstract method to update trainer internal attributes
        """
        raise NotImplementedError


class NNTrainer(Trainer):
    """
    Object that trains neural networks
    """
    def __init__(self, model: Optional[Module], metric: Optional[Callable],
                 lr: Optional[float], batch_size: Optional[int],
                 weight_decay: Optional[float], epochs: int,
                 early_stopping: bool = False, patience: int = 50,
                 device: str = "cpu", in_trial: bool = False):
        """
        Sets protected and public attributes

        Args:
            model: neural network model to train
            metric: function to optimize (not used for back propagation)
            lr: learning rate
            batch_size: size of the batches created by the training data loader
            weight_decay: L2 penalty
            epochs: number of epochs
            early_stopping: True if we want to stop the training when the validation loss stops decreasing
            patience: number of epochs without improvement allowed before early stopping
            device: device used to run our training ("cpu" or "gpu")
            in_trial: True, if the trainer is instantiate from an objective function
        """
        # We make sure that our model is a torch Module
        assert isinstance(model, Module) or model is None, 'model must be a torch.nn.Module'

        # We call parent's constructor
        super().__init__(model=model, metric=metric, device=device)

        # We save public attributes
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience

        # We save protected attribute
        self._in_trial = in_trial
        self._lr = lr
        self._weight_decay = weight_decay

        # We send model to the proper device and set optimizer protected attribute if it is not None
        if model is not None:
            self._optimizer = optim.Adam(params=self._model.parameters(), lr=lr, weight_decay=weight_decay)
            self._model.to(self._device)
        else:
            self._optimizer = None

    def evaluate(self, dataset: PetaleNNDataset, early_stopper: EarlyStopping) -> Tuple[float, float, bool]:
        """
        Calculates the loss on the validation set using a single batch.
        There will be no memory problem since our datasets are really small

        Args:
            early_stopper: early stopping object created in the fit function
            dataset: custom dataset with masks already defined

        Returns: validation epoch loss, validation epoch score and early stopping status

        """

        # Set model for validation
        self._model.eval()

        with torch.no_grad():

            # We extract the continuous data x_cont, the categorical data x_cat
            # and the correct predictions y for the single batch
            inputs, y = self.extract_data(dataset[dataset.valid_mask])

            # We perform the forward pass: compute predicted outputs by passing inputs to the model
            output = self._model(**inputs)

            # We calculate the loss and the score
            val_epoch_loss = self._model.loss(output, y).item()
            score = self._metric(output, y)

        # We early stopping status
        if self.early_stopping:
            early_stopper(val_epoch_loss, self._model)

            if early_stopper.early_stop:
                self._model = early_stopper.get_best_model()
                return val_epoch_loss, score, True

        return val_epoch_loss, score, False

    def extract_data(self, data: Tuple[Optional[tensor], Optional[tensor], tensor]
                     ) -> Tuple[Dict[str, Optional[tensor]], tensor]:
        """
        Extracts data out of a slice of PetaleNNDataset

        Args:
            data: sliced PetaleNNDataset

        Returns: Dict of shape {"x_cont": tensor, "x_cat": tensor} and tensor with targets

        """
        x_cont, x_cat, y = data
        data = {k: v.to(self._device) for k, v in [("x_cont", x_cont), ("x_cat", x_cat)]}
        return data, y.to(self._device)

    def fit(self, dataset: PetaleNNDataset, verbose: bool = True,
            visualization: bool = False, path: Optional[str] = None,
            seed: Optional[int] = None) -> Tuple[tensor, tensor]:
        """
        Fits the model to the given data

        Args:
            dataset: custom dataset with masks already defined
            verbose: True to print progress throughout the epochs
            visualization: True to visualize the progress of the loss in the train and valid set throughout the epochs
            path: emplacement to stores visualization plots
            seed: seed value to have deterministic procedure

        Returns: training losses, validation losses

        """
        assert self._model is not None, "Model must be set before training"
        assert not visualization or path is not None, "Path must be specified"

        # We set the seed value
        if seed is not None:
            manual_seed(seed)

        # We validate the batch size
        training_size = len(dataset.train_mask)
        self.batch_size = min(training_size, self.batch_size)

        # We create the training data loader
        if training_size % self.batch_size == 1:
            train_loader = DataLoader(dataset, batch_size=self.batch_size,
                                      sampler=SubsetRandomSampler(dataset.train_mask), drop_last=True)
        else:
            train_loader = DataLoader(dataset, batch_size=self.batch_size,
                                      sampler=SubsetRandomSampler(dataset.train_mask))

        # We initialize empty lists to store losses and scores
        training_loss, valid_loss, training_score, valid_score = [], [], [], []

        # We init the update function
        update_progress = self.update_progress_func(n_epochs=self.epochs, in_trial=self._in_trial, verbose=verbose)

        # We set an early stopper
        early_stopper = EarlyStopping(self.patience) if self.early_stopping else None

        for epoch in range(self.epochs):

            ###################
            # train the model #
            ###################

            # We calculate training mean epoch loss on all batches
            mean_epoch_loss, train_metric_score = self.train(train_loader)
            update_progress(epoch=epoch, mean_epoch_loss=mean_epoch_loss)

            # We record training loss and score
            training_loss.append(mean_epoch_loss)
            training_score.append(train_metric_score)

            ######################
            # validate the model #
            ######################

            # We calculate validation epoch loss and save it
            val_epoch_loss, val_metric_score, early_stop = self.evaluate(dataset, early_stopper)
            valid_loss.append(val_epoch_loss)
            valid_score.append(val_metric_score)

            if early_stop:
                break

        if visualization:

            # We plot the graph to visualize the training and validation loss
            visualize_epoch_progression(tensor(training_loss), tensor(valid_loss),
                                        progression_type="loss", path=path)

            # We plot the graph to visualize the training and validation metric
            visualize_epoch_progression(tensor(training_score), tensor(valid_score),
                                        progression_type="metric", path=path)
        if self.early_stopping:
            early_stopper.remove_checkpoint()

        return tensor(training_loss), tensor(valid_loss)

    def predict(self, **kwargs) -> tensor:
        """
        Returns log probabilities in the case of an NNClassifier and
        returns real-valued targets in the case of NNRegressor

        Returns: (N, 1) or (N, C) tensor

        """
        return self._model.predict(kwargs['x_cont'], kwargs['x_cat'], log_prob=kwargs['log_prob'])

    def train(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Trains the model for a single epoch

        Args:
            train_loader: training DataLoader

        Returns: mean epoch loss and mean epoch score
        """

        # We set model for training
        self._model.train()
        epoch_loss, epoch_score = 0, 0

        for item in train_loader:

            # We extract the continuous data x_cont, the categorical data x_cat
            # and the correct predictions y
            inputs, y = self.extract_data(item)

            # We clear the gradients of all optimized variables
            self._optimizer.zero_grad()

            # We perform the forward pass: compute predicted outputs by passing inputs to the model
            output = self._model(**inputs)

            # We calculate the loss and the score
            loss = self._model.loss(output, y)
            score = self._metric(output, y)
            epoch_loss += loss.item()
            epoch_score += score

            # We perform the backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # We perform a single optimization step (parameter update)
            self._optimizer.step()

        return epoch_loss / len(train_loader), epoch_score / len(train_loader)

    def update_trainer(self, **kwargs) -> None:
        """
        Updates the model, the weight decay, the batch size, the learning rate and the trial
        """
        new_model = kwargs.get('model', self._model)

        assert isinstance(new_model, Module), 'model argument must inherit from torch.nn.Module'

        # Update of public attributes
        self.batch_size = kwargs.get('batch_size', self.batch_size)

        # Update of protected attributes
        self._model = kwargs.get('model', self._model)
        self._weight_decay = kwargs.get('weight_decay', self._weight_decay)
        self._lr = kwargs.get('lr', self._lr)
        self._optimizer = optim.Adam(params=self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay)

        # We send model to the proper device
        self._model.to(self._device)

    @staticmethod
    def update_progress_func(n_epochs: int, in_trial: bool, verbose: bool) -> Callable:
        """
        Defines a function that updates the training progress

        Args:
            n_epochs: number of epochs to execute
            in_trial: True, if the trainer is instantiate from an objective function
            verbose: True if we want to print progress

        Returns: function

        """
        if not in_trial and verbose:
            def update_progress(epoch: int, mean_epoch_loss: float):
                if (epoch + 1) % 5 == 0 or (epoch + 1) == n_epochs:
                    print(f"Epoch {epoch + 1} - Loss : {round(mean_epoch_loss, 4)}")
        else:
            def update_progress(**kwargs):
                pass

        return update_progress


class RFTrainer(Trainer):
    """
    Object that trains random forest classifier
    """
    EPS = 1*10**(-4)  # constant for log probabilities adjustment

    def __init__(self, model: Optional[RandomForestClassifier], metric: Metric):
        """
        Set protected attributes

        Args:
            model: random forest classifier to train
            metric: score metric to optimize
        """
        super().__init__(model=model, metric=metric)

    def fit(self, dataset: PetaleRFDataset, **kwargs) -> None:
        """
        Trains the classifier

        Args:
            dataset: custom dataset with masks already defined

        Returns: None
        """

        assert self._model is not None, "Model must be set before training"

        x, y = dataset[dataset.train_mask]
        self._model.fit(x, y)

    def predict(self, **kwargs) -> tensor:
        """
        Predicts the log probabilities associated to each class

        Returns: 2D Numpy array (n_samples, n_classes) with log probabilities

        """
        # We compute probabilities
        prob = self._model.predict_proba(kwargs['x'])

        # We make a little adjustment to avoid computation error with log probabilities
        prob[prob == 0] = self.EPS
        prob[prob == 1] = 1 - self.EPS

        # We compute the log of probabilities
        prob = log(prob)

        return tensor(prob)

    def extract_data(self, data: Tuple[DataFrame, array]) -> Tuple[Dict[str, DataFrame], tensor]:
        """
        Simply returns the sliced dataframe in a dictionary

        Args:
            data: sliced PetaleRFDataset

        Returns: x, y
        """
        x, y = data
        return {'x': x}, tensor(y)

    def update_trainer(self, **kwargs) -> None:
        """
        Updates the model
        """
        self._model = kwargs.get('model', self._model)
