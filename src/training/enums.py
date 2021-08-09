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
    ALPHA: str = "alpha"
    BETA: str = "beta"
    LR: str = "lr"
    BATCH_SIZE: str = "batch_size"
    N_LAYERS: str = "n_layers"
    N_UNITS: str = "n_units"
    DROPOUT: str = "dropout"
    ACTIVATION: str = "activation"
    LAYERS: str = "layers"

    def __iter__(self):
        return iter([self.ALPHA, self.BETA, self.LR, self.BATCH_SIZE, self.N_LAYERS,
                     self.N_UNITS, self.DROPOUT, self.ACTIVATION])


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

    def __iter__(self):
        return iter([self.ALPHA, self.BETA])


class Range:
    """
    Stores possible hyperparameters' range types
    """
    MIN: str = "min"
    MAX: str = "max"
    STEP: str = "step"
    VALUES: str = "values"
    VALUE: str = "value"
