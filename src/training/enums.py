"""
This file is used to store useful constants related to models' hyperparameters
and optuna's suggestion functions.
"""

from enum import Enum

# HYPERPARAMETERS IMPORTANCE SEED
HYPER_PARAMS_SEED = 2021


class NeuralNetsHP(Enum):
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


class RandomForestsHP(Enum):
    """
    Stores random forests possible hyperparameters
    """
    N_ESTIMATORS: str = "n_estimators"
    MAX_DEPTH: str = "max_depth"
    MAX_FEATURES: str = "max_features"
    MAX_SAMPLES: str = "max_samples"


class SuggestionFunctions(Enum):
    """
    Stores possible types of suggestion functions
    """
    INT: str = "int"
    UNIFORM: str = "uniform"
    CATEGORICAL: str = "categorical"


class Range(Enum):
    """
    Stores possible hyperparameters' range types
    """
    MIN: str = "min"
    MAX: str = "max"
    VALUES: str = "values"
    VALUE: str = "value"
