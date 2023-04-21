"""
Filename: free_hps.py

Author: Nicolas Raymond

Description: File used to store hps search spaces for the experiments

Date of last modification: 2022/04/12
"""

from src.models.gat import GATHP
from src.models.gcn import GCNHP
from src.models.mlp import MLPHP
from src.models.random_forest import RandomForestHP
from src.models.xgboost_ import XGBoostHP
from src.utils.hyperparameters import Range

RF_HPS = {
    RandomForestHP.MAX_LEAF_NODES.name: {
        Range.MIN: 5,
        Range.MAX: 50,
        Range.STEP: 5,
    },
    RandomForestHP.MAX_FEATURES.name: {
        Range.VALUES: ["sqrt", "log2"],
    },
    RandomForestHP.MAX_SAMPLES.name: {
        Range.MIN: 0.80,
        Range.MAX: 1
    },
    RandomForestHP.MIN_SAMPLES_SPLIT.name: {
        Range.VALUE: 2,
    },
    RandomForestHP.N_ESTIMATORS.name: {
        Range.MIN: 1000,
        Range.MAX: 3000,
        Range.STEP: 250,
    },
}

XGBOOST_HPS = {
    XGBoostHP.ALPHA.name: {
       Range.VALUE: 0
    },
    XGBoostHP.BETA.name: {
        Range.MIN: 0.0005,
        Range.MAX: 1,
    },
    XGBoostHP.LR.name: {
        Range.MIN: 0.005,
        Range.MAX: 0.1
    },
    XGBoostHP.MAX_DEPTH.name: {
        Range.MIN: 1,
        Range.MAX: 3,
    },
    XGBoostHP.SUBSAMPLE.name: {
        Range.MIN: 0.80,
        Range.MAX: 1
    },
}

MLP_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.VALUE: 0
    },
    MLPHP.BATCH_SIZE.name: {
        Range.VALUE: 25
    },
    MLPHP.BETA.name: {
        Range.MIN: 0.0005,
        Range.MAX: 1,
    },
    MLPHP.DROPOUT.name: {
        Range.MIN: 0,
        Range.MAX: 0.50
    },
    MLPHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.1
    },
    MLPHP.RHO.name: {
        Range.VALUE: 0
    },
    MLPHP.N_LAYER.name: {
        Range.VALUE: 1,
    },
    MLPHP.N_UNIT.name: {
        Range.VALUE: 4,
    },
}

ENET_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.VALUE: 0
    },
    MLPHP.BATCH_SIZE.name: {
        Range.VALUE: 25
    },
    MLPHP.BETA.name: {
        Range.MIN: 0.0005,
        Range.MAX: 1,
    },
    MLPHP.DROPOUT.name: {
        Range.VALUE: 0,
    },
    MLPHP.LR.name: {
        Range.MIN: 0.005,
        Range.MAX: 0.1
    },
    MLPHP.RHO.name: {
        Range.VALUE: 0
    },
    MLPHP.N_LAYER.name: {
        Range.VALUE: 0
    },
    MLPHP.N_UNIT.name: {
        Range.VALUE: 5
    },
}

GATHPS = {
    GATHP.ALPHA.name: {
        Range.VALUE: 0
    },
    GATHP.BETA.name: {
        Range.MIN: 0.0005,
        Range.MAX: 1,
    },
    GATHP.FEAT_DROPOUT.name: {
        Range.VALUE: 0,
    },
    GATHP.HIDDEN_SIZE.name: {
        Range.VALUE: None,
    },
    GATHP.LR.name: {
        Range.MIN: 0.005,
        Range.MAX: 0.1
    },
    GATHP.ATTN_DROPOUT.name: {
        Range.VALUE: 0.5
    },
    GATHP.NUM_HEADS.name: {
        Range.VALUE: 10,
    },
    GATHP.RHO.name: {
        Range.VALUE: 0
    },
}

GCNHPS = {
    GCNHP.ALPHA.name: {
        Range.VALUE: 0
    },
    GCNHP.BETA.name: {
        Range.MIN: 0.0005,
        Range.MAX: 1,
    },
    GCNHP.HIDDEN_SIZE.name: {
        Range.VALUE: None,
    },
    GCNHP.LR.name: {
        Range.MIN: 0.005,
        Range.MAX: 0.1
    },
    GCNHP.RHO.name: {
        Range.VALUE: 0
    },
}
