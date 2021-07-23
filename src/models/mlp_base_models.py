"""
Authors: Nicolas Raymond
         Mehdi Mitiche

This file is used to store the MLP with entity embeddings base model and its children PetaleMLPClassifier
and PetaleMLPRegressor
"""

from src.data.processing.datasets import PetaleDataset
from src.training.early_stopping import EarlyStopper
from src.utils.score_metrics import Metric, BalancedAccuracyEntropyRatio, RootMeanSquaredError
from torch import cat, nn, no_grad, tensor, mean, zeros_like, ones, sigmoid
from torch.nn import Module, ModuleList, Embedding,\
    Linear, BatchNorm1d, Dropout, Sequential, BCEWithLogitsLoss, MSELoss
from torch.nn.functional import l1_loss, mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Callable, List, Optional


class MLP(Module):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self, output_size: int, layers: List[int], activation: str,
                 criterion: Callable, criterion_name: str, eval_metric: Metric, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, lr: float = 0.05,  num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None, cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None, verbose: bool = False):

        """
        Builds the layers of the model and sets other protected attributes

        Args:
            output_size: the number of nodes in the last layer of the neural network
            layers: list with number of units in each hidden layer
            criterion: loss function of our model
            criterion_name: name of the loss function of our model
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            lr: learning rate
            num_cont_col: number of numerical continuous columns
                          (equal to number of class in the case of classification)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
        """
        assert num_cont_col is not None or cat_sizes is not None, "There must be continuous columns" \
                                                                  " or categorical columns"
        # We call parent's constructor
        Module.__init__(self)

        # We set protected attributes
        self._alpha = alpha
        self._beta = beta
        self._cat_idx = cat_idx if cat_idx is not None else []
        self._cont_idx = [i for i in range(len(self._cat_idx) + num_cont_col) if i not in self._cat_idx]
        self._criterion = criterion
        self._criterion_name = criterion_name
        self._embedding_layers = None
        self._eval_metric = eval_metric
        self._evaluations = {i: {self._criterion_name: [], self._eval_metric.name: []} for i in ["train", "valid"]}
        self._lr = lr
        self._optimizer = None
        self._verbose = verbose

        # We initialize the input_size
        input_size = num_cont_col if num_cont_col is not None else 0

        # We set the embedding layers
        if cat_idx is not None and cat_sizes is not None:

            # We check embedding sizes (if nothing provided -> emb_sizes = cat_sizes)
            cat_emb_sizes = cat_emb_sizes if cat_emb_sizes is not None else cat_sizes

            # We generate the embedding sizes
            embedding_sizes = [(cat_size, emb_size) for cat_size, emb_size in zip(cat_sizes, cat_emb_sizes)]

            # We create the embedding layers
            self._embedding_layers = ModuleList([Embedding(num_embedding, embedding_dim) for
                                                 num_embedding, embedding_dim in embedding_sizes])
            # We sum the length of all embeddings
            input_size += sum(cat_sizes)

        # We create the different layers of our model
        # Linear --> Activation --> Batch Normalization --> Dropout
        all_layers = []
        for i in layers:
            all_layers.append(Linear(input_size, i))
            all_layers.append(getattr(nn, activation)())
            all_layers.append(BatchNorm1d(i))
            all_layers.append(Dropout(dropout))
            input_size = i

        # We define the output layer
        if len(layers) == 0:
            all_layers.append(Linear(input_size, output_size))
        else:
            all_layers.append(Linear(layers[-1], output_size))

        # We save all our layers in self.layers
        self._layers = Sequential(*all_layers)

    def _execute_train_step(self, train_loader: DataLoader, sample_weights: tensor) -> float:
        """
        Executes one training epoch

        Args:
            train_loader: training data loader

        Returns: mean epoch loss
        """
        # We set the model for training
        self.train()
        epoch_loss, epoch_score = 0, 0

        # We execute one training step
        for item in train_loader:

            # We extract the data
            x, y, idx = item

            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the forward pass
            output = self(x)

            # We calculate the loss and the score
            loss = self.loss(sample_weights[idx], output, y)
            score = self._eval_metric(output, y)
            epoch_loss += loss.item()
            epoch_score += score

            # We perform the backward pass
            loss.backward()

            # We perform a single optimization step (parameter update)
            self._optimizer.step()

        # We save mean epoch loss and mean epoch score
        nb_batch = len(train_loader)
        mean_epoch_loss = epoch_loss/nb_batch
        self._evaluations["train"][self._criterion_name].append(mean_epoch_loss)
        self._evaluations["train"][self._eval_metric.name].append(epoch_score/nb_batch)

        return mean_epoch_loss

    def _execute_valid_step(self, valid_loader: Optional[DataLoader], early_stopper: EarlyStopper) -> bool:
        """
        Executes an inference step on the validation data

        Args:
            valid_loader: validation data loader
            early_stopper: early stopper keeping track of validation loss

        Returns: True if we need to early stop
        """
        if valid_loader is None:
            return False

        # Set model for evaluation
        self.eval()
        epoch_loss, epoch_score = 0, 0

        # We execute one inference step on validation set
        with no_grad():

            for item in valid_loader:

                # We extract the data
                x, y, idx = item

                # We perform the forward pass: compute predicted outputs by passing inputs to the model
                output = self(x)

                # We calculate the loss and the score
                batch_size = len(idx)
                sample_weights = ones(batch_size)/batch_size
                loss = self.loss(sample_weights, output, y)  # Sample weights are equal for validation (1/N)
                score = self._eval_metric(output, y)
                epoch_loss += loss.item()
                epoch_score += score

        # We save mean epoch loss and mean epoch score
        nb_batch = len(valid_loader)
        mean_epoch_loss = epoch_loss / nb_batch
        self._evaluations["valid"][self._criterion_name].append(mean_epoch_loss)
        self._evaluations["valid"][self._eval_metric.name].append(epoch_score / nb_batch)

        # We early stopping status
        early_stopper(epoch_loss, self)

        if early_stopper.early_stop:
            self.load_state_dict(early_stopper.get_best_params())
            return True

        return False

    def _generate_progress_func(self, max_epochs: int) -> Callable:
        """
        Defines a function that updates the training progress

        Args:
            max_epochs: maximum number of training epochs

        Returns: function
        """
        if self._verbose:
            def update_progress(epoch: int, mean_epoch_loss: float):
                if (epoch + 1) % 5 == 0 or (epoch + 1) == max_epochs:
                    print(f"Epoch {epoch + 1} - Loss : {round(mean_epoch_loss, 4)}")
        else:
            def update_progress(*args):
                pass

        return update_progress

    def fit(self, dataset: PetaleDataset, batch_size: int = 55,
            valid_batch_size: Optional[int] = None, max_epochs: int = 200, patience: int = 15,
            sample_weights: Optional[tensor] = None) -> None:
        """
        Fits the model to the training data

        Args:
            dataset: PetaleDataset used to feed data loaders
            batch_size: size of the batches in the training loader
            valid_batch_size: size of the batches in the valid loader (None = one single batch)
            max_epochs: Maximum number of epochs for training
            patience: Number of consecutive epochs without improvement
            sample_weights: (N,) tensor with weights of the samples in the training set

        Returns: None
        """
        # We check the validity of the samples' weights
        dataset_size = len(dataset)
        if sample_weights is not None:
            assert(sample_weights.shape[0] == dataset_size), f"Sample weights as length {sample_weights.shape[0]}" \
                                                              f" while dataset as length {dataset_size}"
        else:
            sample_weights = ones(dataset_size)/dataset_size

        # We create the training data loader
        train_loader = DataLoader(dataset, batch_size=min(len(dataset.train_mask), batch_size),
                                  sampler=SubsetRandomSampler(dataset.train_mask))

        # We create the valid data loader
        valid_size, valid_loader, = len(dataset.valid_mask), None
        early_stopper, early_stopping = None, False
        if valid_size != 0:
            valid_bs = valid_batch_size if valid_batch_size is not None else valid_size
            valid_loader = DataLoader(dataset, batch_size=min(valid_size, valid_bs),
                                      sampler=SubsetRandomSampler(dataset.valid_mask))
            early_stopper, early_stopping = EarlyStopper(patience), True

        # We init the update function
        update_progress = self._generate_progress_func(max_epochs)

        # We set the optimizer
        self._optimizer = Adam(self.parameters(), lr=self._lr)

        # We execute the epochs
        for epoch in range(max_epochs):

            # We calculate training mean epoch loss on all batches
            mean_epoch_loss = self._execute_train_step(train_loader, sample_weights)
            update_progress(epoch, mean_epoch_loss)

            # We proceed to calculate valid mean epoch loss and apply early stopping if needed
            if self._execute_valid_step(valid_loader, early_stopper):
                break

        if early_stopping:
            early_stopper.remove_checkpoint()

    def forward(self, x: tensor) -> tensor:
        """
        Executes the forward pass

        Args:
            x: (N,D) tensor with D-dimensional samples

        Returns: tensor with values of the node from the last layer

        """
        # We initialize list of tensors to concatenate
        new_x = []

        # We extract continuous data
        if len(self._cont_idx) != 0:
            new_x.append(x[:, self._cont_idx])

        # We perform entity embeddings
        if len(self._cat_idx) != 0:
            x_cat = x[:, self._cat_idx]
            embeddings = []
            for i, e in enumerate(self._embedding_layers):
                embeddings.append(e(x_cat[:, i].long()))

            # We concatenate all the embeddings
            new_x.append(cat(embeddings, 1))

        # We concatenate all inputs
        x = cat(new_x, 1)

        return self._layers(x).squeeze()

    def loss(self, sample_weights: tensor, pred: tensor, y: tensor) -> tensor:
        """
        Calls the criterion and add elastic penalty

        Args:
            sample_weights: (N,) tensor with weights of samples on which we calculate loss
            pred: (N, C) tensor if classification with C classes, (N,) tensor for regression
            y: (N,) tensor with targets

        Returns: tensor with loss value
        """
        # Computations of penalties
        flatten_params = [w.view(-1, 1) for w in self.parameters()]
        l1_penalty = mean(tensor([l1_loss(w, zeros_like(w)) for w in flatten_params]))
        l2_penalty = mean(tensor([mse_loss(w, zeros_like(w)) for w in flatten_params]))

        # Computation of loss without reduction
        loss = self._criterion(pred, y.float())  # (N,) tensor

        # Computation of loss reduction + elastic penalty
        return (loss * sample_weights / sample_weights.sum()).sum() + self._alpha*l1_penalty + self._beta*l2_penalty


class MLPBinaryClassifier(MLP):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self, layers: List[int], activation: str,
                 eval_metric: Optional[Metric] = None, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, lr: float = 0.05, num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None, cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None, verbose: bool = False):
        """
        Sets protected attributes using parent's constructor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            lr: learning rate
            num_cont_col: number of numerical continuous columns
                          (equal to number of class in the case of classification)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
        """
        eval_metric = eval_metric if eval_metric is not None else BalancedAccuracyEntropyRatio()
        super().__init__(output_size=1, layers=layers, activation=activation,
                         criterion=BCEWithLogitsLoss(reduction='none'), criterion_name='WBCE',
                         eval_metric=eval_metric, dropout=dropout, alpha=alpha, beta=beta,
                         lr=lr, num_cont_col=num_cont_col, cat_idx=cat_idx, cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes, verbose=verbose)

    def predict_proba(self, x: tensor) -> tensor:
        """
        Predict classes' probabilities for all samples

        Args:
            x: (N,D) tensor with D-dimensional samples


        Returns: (N, C) tensor where C is the number of classes
        """
        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a softmax
        return sigmoid(self(x))


class MLPRegressor(MLP):
    """
    Multilayer perceptron model with entity embedding
    """
    def __init__(self, layers: List[int], activation: str,
                 eval_metric: Optional[Metric] = None, dropout: float = 0,
                 alpha: float = 0, beta: float = 0, lr: float = 0.05, num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None, cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None, verbose: bool = False):
        """
        Sets protected attributes using parent's constructor

        Args:
            layers: list with number of units in each hidden layer
            eval_metric: evaluation metric
            activation: activation function
            dropout: probability of dropout
            alpha: L1 penalty coefficient
            beta: L2 penalty coefficient
            lr: learning rate
            num_cont_col: number of numerical continuous columns
                          (equal to number of class in the case of classification)
            cat_idx: idx of categorical columns in the dataset
            cat_sizes: list of integer representing the size of each categorical column
            cat_emb_sizes: list of integer representing the size of each categorical embedding
        """
        eval_metric = eval_metric if eval_metric is not None else RootMeanSquaredError()
        super().__init__(output_size=1, layers=layers, activation=activation,
                         criterion=MSELoss(reduction='none'), criterion_name='MSE',
                         eval_metric=eval_metric, dropout=dropout, alpha=alpha, beta=beta,
                         lr=lr, num_cont_col=num_cont_col, cat_idx=cat_idx, cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes, verbose=verbose)

    def predict(self, x: tensor) -> tensor:
        """
        Returns the predicted real-valued targets for all samples

        Args:
            x: (N,D) tensor or array with D-dimensional samples

        Returns: (N,) tensor or array
        """
        # Set model for evaluation
        self.eval()

        # Execute a forward pass and apply a softmax
        return self(x)
