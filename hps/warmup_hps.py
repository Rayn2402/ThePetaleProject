"""
 File used to store hyperparameters used for evaluations
"""

from src.training.enums import *

NN_HPS = {
    NeuralNetsHP.LR: {
        Range.MIN: 1e-3,
        Range.MAX: 1e-1
    },
    NeuralNetsHP.BATCH_SIZE: {
        Range.MIN: 5,
        Range.MAX: 50
    },
    NeuralNetsHP.N_LAYERS: {
        Range.MIN: 1,
        Range.MAX: 3,
    },
    NeuralNetsHP.N_UNITS: {
        Range.MIN: 1,
        Range.MAX: 20,
    },
    NeuralNetsHP.DROPOUT: {
        Range.VALUE: 0
    },
    NeuralNetsHP.ACTIVATION: {
        Range.VALUE: "ReLU"
    },
    NeuralNetsHP.WEIGHT_DECAY: {
        Range.MIN: 1e-8,
        Range.MAX: 1e-1
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

ELASTIC_HPS = {
    ElasticNetHP.BETA: {
        Range.MIN: 1e-8,
        Range.MAX: 5
    },
    ElasticNetHP.ALPHA: {
        Range.MIN: 1e-8,
        Range.MAX: 5
    }
}
