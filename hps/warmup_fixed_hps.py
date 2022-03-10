"""
Filename: warmup_fixed_hps.py

Author: Nicolas Raymond

Description: File used to store fixed hps for warmup experiments

Date of last modification: 2022/03/01
"""

from src.models.gat import GATHP
from src.models.gge import GGEHP
from src.models.mlp import MLPHP
from src.models.random_forest import RandomForestHP
from src.models.xgboost_ import XGBoostHP
from src.utils.hyperparameters import Range

RF_HPS = {
    RandomForestHP.MAX_LEAF_NODES.name: {
        Range.VALUE: 25,
    },
    RandomForestHP.MAX_FEATURES.name: {
        Range.VALUE: "sqrt",
    },
    RandomForestHP.MAX_SAMPLES.name: {
        Range.VALUE: 0.80,
    },
    RandomForestHP.MIN_SAMPLES_SPLIT.name: {
        Range.VALUE: 2,
    },
    RandomForestHP.N_ESTIMATORS.name: {
        Range.VALUE: 2000,
    },
}

XGBOOST_HPS = {
    XGBoostHP.ALPHA.name: {
        Range.VALUE: 0,
    },
    XGBoostHP.BETA.name: {
        Range.VALUE: .0005
    },
    XGBoostHP.LR.name: {
        Range.VALUE: 0.001,
    },
    XGBoostHP.MAX_DEPTH.name: {
        Range.VALUE: 3,
    },
    XGBoostHP.SUBSAMPLE.name: {
        Range.VALUE: 0.80,
    },
}

MLP_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.VALUE: 0,
    },
    MLPHP.BATCH_SIZE.name: {
        Range.VALUE: 5,
    },
    MLPHP.BETA.name: {
        Range.VALUE: 0.0005
    },
    MLPHP.DROPOUT.name: {
        Range.VALUE: 0.25,
    },
    MLPHP.LR.name: {
        Range.VALUE: 0.001,
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
        Range.VALUE: 0,
    },
    MLPHP.BATCH_SIZE.name: {
        Range.VALUE: 5,
    },
    MLPHP.BETA.name: {
        Range.VALUE: 0.0005
    },
    MLPHP.DROPOUT.name: {
        Range.VALUE: 0,
    },
    MLPHP.LR.name: {
        Range.VALUE: 0.001,
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

ENET_GGE_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.VALUE: 0,
    },
    MLPHP.BATCH_SIZE.name: {
        Range.VALUE: 5,
    },
    MLPHP.BETA.name: {
        Range.VALUE: 0.0005
    },
    MLPHP.DROPOUT.name: {
        Range.VALUE: 0.25
    },
    MLPHP.LR.name: {
        Range.VALUE: 0.001,
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
        Range.VALUE: 0,
    },
    GATHP.BATCH_SIZE.name: {
        Range.VALUE: 5,
    },
    GATHP.BETA.name: {
        Range.MAX: 0.0005
    },
    GATHP.FEAT_DROPOUT.name: {
        Range.VALUE: 0.25,
    },
    GATHP.HIDDEN_SIZE.name: {
        Range.VALUE: 8,
    },
    GATHP.LR.name: {
        Range.VALUE: 0.001,
    },
    GATHP.ATTN_DROPOUT.name: {
        Range.VALUE: 0.25,
    },
    GATHP.NUM_HEADS.name: {
        Range.VALUE: 1,
    },
    GATHP.RHO.name: {
        Range.VALUE: 0
    },
}
GGEHPS = {
    GGEHP.BATCH_SIZE.name: {
        Range.VALUE: 5,
    },
    GGEHP.DROPOUT.name: {
        Range.VALUE: 0.25,
    },
    GGEHP.LR.name: {
        Range.VALUE: 0.001,
    },
    GGEHP.RHO.name: {
        Range.MIN: 0.05,
    }
}
