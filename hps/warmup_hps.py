"""
 File used to store hyperparameters used for evaluations
"""

from src.training.enums import *

NN_ENET_HPS = {
    NeuralNetsHP.ALPHA: {
        Range.MIN: 1e-8,
        Range.MAX: 5,
    },
    NeuralNetsHP.BETA: {
        Range.MIN: 1e-8,
        Range.MAX: 5,
    },
    NeuralNetsHP.LR: {
        Range.MIN: 1e-3,
        Range.MAX: 1e-1
    },
    NeuralNetsHP.BATCH_SIZE: {
        Range.MIN: 5,
        Range.MAX: 50
    },
    NeuralNetsHP.N_LAYERS: {
        Range.VALUE: 0
    },
    NeuralNetsHP.N_UNITS: {
        Range.VALUE: 0
    },
    NeuralNetsHP.DROPOUT: {
        Range.VALUE: 0
    },
    NeuralNetsHP.ACTIVATION: {
        Range.VALUE: "ReLU"
    },
}

NN_LOW_HPS = {
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
        Range.MAX: 5,
    },
    NeuralNetsHP.N_UNITS: {
        Range.MIN: 1,
        Range.MAX: 20,
    },
    NeuralNetsHP.DROPOUT: {
        Range.VALUE: 0
    },
    NeuralNetsHP.ACTIVATION: {
        Range.VALUES: ["ReLU", "PReLU"]
    },
}

NN_HIGH_HPS = {
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
        Range.MAX: 2,
    },
    NeuralNetsHP.N_UNITS: {
        Range.MIN: 1,
        Range.MAX: 10,
    },
    NeuralNetsHP.DROPOUT: {
        Range.VALUE: 0
    },
    NeuralNetsHP.ACTIVATION: {
        Range.VALUE: "ReLU"
    },
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
