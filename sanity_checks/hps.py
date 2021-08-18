"""
 File used to store hyperparameters used for sanity checks
"""

from src.models.mlp import MLPHP
from src.models.han import HanHP
from src.models.tabnet import TabNetHP
from src.models.random_forest import RandomForestHP
from src.models.xgboost_ import XGBoostHP
from src.training.enums import Range

TAB_HPS = {
    TabNetHP.BATCH_SIZE.name: {
        Range.MIN: 25,
        Range.MAX: 50,
        Range.STEP: 5
    },
    TabNetHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
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
        Range.MIN: 3,
        Range.MAX: 10
    },
    TabNetHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
    TabNetHP.WEIGHT.name: {
        Range.VALUE: 0.50,
    },
}

RF_HPS = {
    RandomForestHP.MAX_DEPTH.name: {
        Range.MIN: 5,
        Range.MAX: 20
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
    RandomForestHP.WEIGHT.name: {
        Range.VALUE: 0.50,
    },
}

XGBOOST_HPS = {
    XGBoostHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    XGBoostHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
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
        Range.MIN: 0.65,
        Range.MAX: 1
    },
    XGBoostHP.WEIGHT.name: {
        Range.VALUE: 0.50,
    },
}

HAN_HPS = {
    HanHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    HanHP.BATCH_SIZE.name: {
        Range.MIN: 25,
        Range.MAX: 50,
        Range.STEP: 5
    },
    HanHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    HanHP.HIDDEN_SIZE.name: {
        Range.MIN: 5,
        Range.MAX: 25,
        Range.STEP: 5
    },
    HanHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
    HanHP.NUM_HEADS.name: {
        Range.MIN: 2,
        Range.MAX: 10
    },
    HanHP.WEIGHT.name: {
        Range.VALUE: 0.50,
    },
}

MLP_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    MLPHP.BATCH_SIZE.name: {
        Range.MIN: 25,
        Range.MAX: 50,
        Range.STEP: 5
    },
    MLPHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    MLPHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
    MLPHP.N_LAYER.name: {
        Range.MIN: 1,
        Range.MAX: 10
    },
    MLPHP.N_UNIT.name: {
        Range.MIN: 2,
        Range.MAX: 10
    },
    MLPHP.WEIGHT.name: {
        Range.VALUE: 0.50,
    },
}
