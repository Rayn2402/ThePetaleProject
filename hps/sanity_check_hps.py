"""
File used to store hyperparameters used for sanity checks
"""

from src.models.mlp import MLPHP
from src.models.han import HanHP
from src.models.tabnet import TabNetHP
from src.models.random_forest import RandomForestHP
from src.models.xgboost_ import XGBoostHP
from src.utils.hyperparameters import Range


TAB_HPS = {
    TabNetHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    TabNetHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 2
    },
    TabNetHP.GAMMA.name: {
        Range.MIN: 1,
        Range.MAX: 2
    },
    TabNetHP.N_A.name: {
        Range.MIN: 2,
        Range.MAX: 10
    },
    TabNetHP.N_D.name: {
        Range.MIN: 2,
        Range.MAX: 10
    },
    TabNetHP.N_STEPS.name: {
        Range.MIN: 1,
        Range.MAX: 5
    },
    TabNetHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
}

RF_HPS = {
    RandomForestHP.MAX_LEAF_NODES.name: {
        Range.MIN: 5,
        Range.MAX: 25,
        Range.STEP: 5,
    },
    RandomForestHP.MAX_FEATURES.name: {
        Range.VALUES: ["sqrt", "log2"],
    },
    RandomForestHP.MAX_SAMPLES.name: {
        Range.MIN: 0.65,
        Range.MAX: 1
    },
    RandomForestHP.MIN_SAMPLES_SPLIT.name: {
        Range.MIN: 2,
        Range.MAX: 5,
    },
    RandomForestHP.N_ESTIMATORS.name: {
        Range.MIN: 1000,
        Range.MAX: 3000,
        Range.STEP: 250,
    },
}

XGBOOST_HPS = {
    XGBoostHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 2
    },
    XGBoostHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 2
    },
    XGBoostHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
    XGBoostHP.MAX_DEPTH.name: {
        Range.MIN: 1,
        Range.MAX: 10,
    },
    XGBoostHP.SUBSAMPLE.name: {
        Range.VALUE: 1,
    },
}

HAN_HPS = {
    HanHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 2
    },
    HanHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    HanHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 2
    },
    HanHP.DROPOUT.name: {
        Range.VALUE: 0,
    },
    HanHP.HIDDEN_SIZE.name: {
        Range.MIN: 2,
        Range.MAX: 10,
        Range.STEP: 2
    },
    HanHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
    HanHP.RHO.name: {
        Range.VALUE: 0
    },
    HanHP.NUM_HEADS.name: {
        Range.MIN: 5,
        Range.MAX: 20
    },
}

MLP_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 2
    },
    MLPHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    MLPHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 2
    },
    MLPHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
    MLPHP.RHO.name: {
        Range.VALUE: 0
    },
    MLPHP.N_LAYER.name: {
        Range.MIN: 1,
        Range.MAX: 3
    },
    MLPHP.N_UNIT.name: {
        Range.MIN: 2,
        Range.MAX: 10
    },
}

ENET_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 3
    },
    MLPHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    MLPHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 3
    },
    MLPHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
    MLPHP.N_LAYER.name: {
        Range.VALUE: 0
    },
    MLPHP.N_UNIT.name: {
        Range.VALUE: 5
    },
}