"""
 File used to store hyperparameters used for sanity checks
"""

from src.training.constants import *

NN_HPS = {
    LR: {
        MIN: 1e-2,
        MAX: 1e-1
    },
    BATCH_SIZE: {
        VALUE: 50
    },
    N_LAYERS: {
        MIN: 1,
        MAX: 3,
    },
    N_UNITS: {
        MIN: 2,
        MAX: 5,
    },
    DROPOUT: {
        VALUE: 0
    },
    ACTIVATION: {
        VALUE: "ReLU"
    },
    WEIGHT_DECAY: {
        VALUE: 0.1
    }
}

RF_HPS = {
    N_ESTIMATORS: {
        MIN: 80,
        MAX: 120,
    },
    MAX_FEATURES: {
        MIN: .8,
        MAX: 1,
    },
    MAX_SAMPLES: {
        MIN: .6,
        MAX: .8,
    },
    MAX_DEPTH: {
        VALUE: 50
    }
}
