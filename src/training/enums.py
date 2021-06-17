"""
This file is used to store useful constants related to models' hyperparameters
and optuna's suggestion functions.
"""

# HYPERPARAMETERS IMPORTANCE SEED
HYPER_PARAMS_SEED = 2021


class NeuralNetsHP:
    """
    Stores neural networks possible hyperparameters
    """
    LR: str = "lr"
    BATCH_SIZE: str = "batch_size"
    N_LAYERS: str = "n_layers"
    N_UNITS: str = "n_units"
    DROPOUT: str = "dropout"
    ACTIVATION: str = "activation"
    WEIGHT_DECAY: str = "weight_decay"
    LAYERS: str = "layers"

    def __iter__(self):
        return iter([self.LR, self.BATCH_SIZE, self.N_LAYERS, self.N_UNITS,
                     self.DROPOUT, self.ACTIVATION, self.WEIGHT_DECAY])


class RandomForestsHP:
    """
    Stores random forests possible hyperparameters
    """
    N_ESTIMATORS: str = "n_estimators"
    MAX_DEPTH: str = "max_depth"
    MAX_FEATURES: str = "max_features"
    MAX_SAMPLES: str = "max_samples"

    def __iter__(self):
        return iter([self.N_ESTIMATORS, self.MAX_DEPTH, self.MAX_FEATURES, self.MAX_SAMPLES])


class ElasticNetHP:
    """
    Stores ElasticNet possible hyperparameters
    """
    ALPHA: str = "alpha"
    BETA: str = "beta"
    DEGREE: str = "degree"


class SuggestFunctions:
    """
    Stores possible types of suggestion functions
    """
    INT: str = "int"
    UNIFORM: str = "uniform"
    CATEGORICAL: str = "categorical"


class Range:
    """
    Stores possible hyperparameters' range types
    """
    MIN: str = "min"
    MAX: str = "max"
    VALUES: str = "values"
    VALUE: str = "value"
