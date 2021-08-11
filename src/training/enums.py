"""
This file is used to store useful constants related to models' hyperparameters
and optuna's suggestion functions.
"""

# HYPERPARAMETERS IMPORTANCE SEED
HYPER_PARAMS_SEED = 2021


class Range:
    """
    Stores possible hyperparameters' range types
    """
    MIN: str = "min"
    MAX: str = "max"
    STEP: str = "step"
    VALUES: str = "values"
    VALUE: str = "value"
