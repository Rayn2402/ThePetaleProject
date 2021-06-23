"""
 File used to store hyperparameters used for sanity checks
"""

from src.training.enums import *

NN_HPS = {
    NeuralNetsHP.LR: {
        Range.MIN: 1e-2,
        Range.MAX: 1e-1
    },
    NeuralNetsHP.BATCH_SIZE: {
        Range.VALUE: 50
    },
    NeuralNetsHP.N_LAYERS: {
        Range.MIN: 1,
        Range.MAX: 3,
    },
    NeuralNetsHP.N_UNITS: {
        Range.MIN: 2,
        Range.MAX: 5,
    },
    NeuralNetsHP.DROPOUT: {
        Range.VALUE: 0
    },
    NeuralNetsHP.ACTIVATION: {
        Range.VALUE: "ReLU"
    },
    NeuralNetsHP.WEIGHT_DECAY: {
        Range.VALUE: 0.1
    }
}

RF_HPS = {
    RandomForestsHP.N_ESTIMATORS: {
        Range.MIN: 80,
        Range.MAX: 120,
    },
    RandomForestsHP.MAX_FEATURES: {
        Range.MIN: .8,
        Range.MAX: 1,
    },
    RandomForestsHP.MAX_SAMPLES: {
        Range.MIN: .6,
        Range.MAX: .8,
    },
    RandomForestsHP.MAX_DEPTH: {
        Range.VALUE: 50
    }
}
