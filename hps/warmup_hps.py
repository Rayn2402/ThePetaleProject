"""
Filename: warmup_hps.py

Author: Nicolas Raymond

Description: File used to store hps search spaces for warmup experiments

Date of last modification: 2022/03/01
"""

from src.models.gat import GATHP
from src.models.gge import GGEHP
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
        Range.MAX: 0.1
    },
}

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
        Range.MIN: 0,
        Range.MAX: 1
    },
    XGBoostHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    XGBoostHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.1
    },
    XGBoostHP.MAX_DEPTH.name: {
        Range.MIN: 1,
        Range.MAX: 10,
    },
    XGBoostHP.SUBSAMPLE.name: {
        Range.MIN: 0.80,
        Range.MAX: 1
    },
}

HAN_HPS = {
    HanHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    HanHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    HanHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
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
        Range.MAX: 1
    },
    MLPHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    MLPHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    MLPHP.DROPOUT.name: {
        Range.MIN: 0,
        Range.MAX: 0.25
    },
    MLPHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.1
    },
    MLPHP.RHO.name: {
        Range.VALUE: 0
    },
    MLPHP.N_LAYER.name: {
        Range.MIN: 1,
        Range.MAX: 5
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
        Range.MAX: 1
    },
    MLPHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    MLPHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    MLPHP.DROPOUT.name: {
        Range.VALUE: 0,
    },
    MLPHP.LR.name: {
        Range.MIN: 0.001,
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

ENET_GGE_HPS = {
    MLPHP.ACTIVATION.name: {
        Range.VALUE: "PReLU"
    },
    MLPHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    MLPHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    MLPHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    MLPHP.DROPOUT.name: {
        Range.MIN: 0,
        Range.MAX: 0.20
    },
    MLPHP.LR.name: {
        Range.MIN: 0.001,
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
        Range.MIN: 0,
        Range.MAX: 1
    },
    GATHP.BATCH_SIZE.name: {
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    GATHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 1
    },
    GATHP.FEAT_DROPOUT.name: {
        Range.VALUE: 0.20
    },
    GATHP.HIDDEN_SIZE.name: {
        Range.MIN: 2,
        Range.MAX: 10,
        Range.STEP: 2
    },
    GATHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.1
    },
    GATHP.ATTN_DROPOUT.name: {
        Range.VALUE: 0.5
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
        Range.MIN: 15,
        Range.MAX: 55,
        Range.STEP: 10
    },
    GGEHP.DROPOUT.name: {
        Range.MIN: 0,
        Range.MAX: 0.25,
    },
    GGEHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.1
    },
    GGEHP.RHO.name: {
        Range.MIN: 0.05,
        Range.MAX: 2
    }
}
