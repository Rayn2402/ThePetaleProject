"""
Filename: fixed_hps.py

Author: Nicolas Raymond

Description: File used to store fixed hps for warmup experiments

Date of last modification: 2022/03/28
"""

from src.models.gat import GATHP
from src.models.gcn import GCNHP
from src.models.gge import GGEHP
from src.models.mlp import MLPHP
from src.models.random_forest import RandomForestHP
from src.models.xgboost_ import XGBoostHP

RF_HPS = {
    RandomForestHP.MAX_LEAF_NODES.name: 25,
    RandomForestHP.MAX_FEATURES.name: "sqrt",
    RandomForestHP.MAX_SAMPLES.name: 0.80,
    RandomForestHP.MIN_SAMPLES_SPLIT.name: 2,
    RandomForestHP.N_ESTIMATORS.name: 2000,
}

XGBOOST_HPS = {
    XGBoostHP.ALPHA.name: 0,
    XGBoostHP.BETA.name: .0005,
    XGBoostHP.LR.name: 0.1,
    XGBoostHP.MAX_DEPTH.name: 2,
    XGBoostHP.SUBSAMPLE.name: 0.80,
}

MLP_HPS = {
    MLPHP.ACTIVATION.name: "PReLU",
    MLPHP.ALPHA.name: 0,
    MLPHP.BATCH_SIZE.name: 5,
    MLPHP.BETA.name: 0.0005,
    MLPHP.DROPOUT.name: 0.25,
    MLPHP.LR.name: 0.001,
    MLPHP.RHO.name: 0,
    MLPHP.N_LAYER.name: 1,
    MLPHP.N_UNIT.name: 4,
}

ENET_HPS = {
    MLPHP.ACTIVATION.name: "PReLU",
    MLPHP.ALPHA.name: 0,
    MLPHP.BATCH_SIZE.name: 5,
    MLPHP.BETA.name: 0.0005,
    MLPHP.DROPOUT.name: 0,
    MLPHP.LR.name: 0.01,
    MLPHP.RHO.name: 0,
    MLPHP.N_LAYER.name: 0,
    MLPHP.N_UNIT.name: 5
}

ENET_GGE_HPS = {
    MLPHP.ACTIVATION.name: "PReLU",
    MLPHP.ALPHA.name: 0,
    MLPHP.BATCH_SIZE.name: 5,
    MLPHP.BETA.name: 0.0005,
    MLPHP.DROPOUT.name: 0.25,
    MLPHP.LR.name: 0.01,
    MLPHP.RHO.name: 0,
    MLPHP.N_LAYER.name: 0,
    MLPHP.N_UNIT.name: 5
}

GATHPS = {
    GATHP.ALPHA.name: 0,
    GATHP.BETA.name: 0.005,
    GATHP.FEAT_DROPOUT.name: 0,
    GATHP.HIDDEN_SIZE.name: None,
    GATHP.LR.name: 0.01,
    GATHP.ATTN_DROPOUT.name: 0.50,
    GATHP.NUM_HEADS.name: 1,
    GATHP.RHO.name: 0,
}

GCNHPS = {
    GCNHP.ALPHA.name: 0,
    GCNHP.BETA.name: 0.005,
    GCNHP.HIDDEN_SIZE.name: None,
    GCNHP.LR.name: 0.01,
    GCNHP.RHO.name: 0,
}

GGEHPS = {
    GGEHP.BATCH_SIZE.name: 5,
    GGEHP.BETA.name: 0.0005,
    GGEHP.DROPOUT.name: 0.25,
    GGEHP.LR.name: 0.005,
}
