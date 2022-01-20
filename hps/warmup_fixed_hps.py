"""
Filename: warmup_fixed_hps.py

Author: Nicolas Raymond

Description: File used to store best preforming hps for each model and each experiment

Date of last modification: 2022/01/19
"""

from src.models.han import HanHP
from src.models.mlp import MLPHP
from src.models.random_forest import RandomForestHP
from src.models.tabnet import TabNetHP
from src.models.xgboost_ import XGBoostHP

# Parameter keys
B = 'baselines'
S = 'sex'
G = 'genes'
GS = 'genomic_signature'
P = 'pretraining'
ES = 'embedding_sharing'

# Model keys
T = 'TabNet'
RF = 'RandomForest'
XG = 'XGBoost'
MLP = 'MLP'
E = 'ENet'
HAN = 'HAN'


FIXED_HPS = {(B,): {
                    T: {
                       TabNetHP.BATCH_SIZE: 15,
                       TabNetHP.BETA: 0.10656,
                       TabNetHP.GAMMA: 1.978489,
                       TabNetHP.N_A: 3,
                       TabNetHP.N_D: 7,
                       TabNetHP.N_STEPS: 5,
                       TabNetHP.LR: 0.05207
                    },
                    RF: {
                        RandomForestHP.MAX_LEAF_NODES: 25,
                        RandomForestHP.MAX_FEATURES: "sqrt",
                        RandomForestHP.MAX_SAMPLES: 0.922954,
                        RandomForestHP.MIN_SAMPLES_SPLIT: 4,
                        RandomForestHP.N_ESTIMATORS: 1250
                    },
                    XG: {
                        XGBoostHP.ALPHA: 1.240812,
                        XGBoostHP.BETA: 0.968769,
                        XGBoostHP.LR: 0.250774,
                        XGBoostHP.MAX_DEPTH: 1,
                        XGBoostHP.SUBSAMPLE: 1,
                    },
                    MLP: {
                        MLPHP.ACTIVATION: "PReLU",
                        MLPHP.ALPHA: 0.672994,
                        MLPHP.BATCH_SIZE: 55,
                        MLPHP.BETA: 0.729828,
                        MLPHP.LR: 0.027521,
                        MLPHP.N_LAYER: 1,
                        MLPHP.N_UNIT: 5,
                        MLPHP.RHO: 0.72377
                    },
                    E: {
                        MLPHP.ACTIVATION: "PReLU",
                        MLPHP.ALPHA: 2.598665,
                        MLPHP.BATCH_SIZE: 35,
                        MLPHP.BETA: 2.759232,
                        MLPHP.LR: 0.462388,
                        MLPHP.N_LAYER: 0,
                        MLPHP.N_UNIT: 5,
                        MLPHP.RHO: 0.73456

                    },
                   },
             (B, S): {
                      T: {
                         TabNetHP.BATCH_SIZE: 15,
                         TabNetHP.BETA: 0.07302,
                         TabNetHP.GAMMA: 1.78896,
                         TabNetHP.N_A: 6,
                         TabNetHP.N_D: 6,
                         TabNetHP.N_STEPS: 5,
                         TabNetHP.LR: 0.02654
                      },
                      RF: {
                         RandomForestHP.MAX_LEAF_NODES: 25,
                         RandomForestHP.MAX_FEATURES: "log2",
                         RandomForestHP.MAX_SAMPLES: 0.968991,
                         RandomForestHP.MIN_SAMPLES_SPLIT: 3,
                         RandomForestHP.N_ESTIMATORS: 2000
                      },
                      XG: {
                          XGBoostHP.ALPHA: 1.821231,
                          XGBoostHP.BETA: 1.714922,
                          XGBoostHP.LR: 0.116989,
                          XGBoostHP.MAX_DEPTH: 1,
                          XGBoostHP.SUBSAMPLE: 1

                      },
                      MLP: {
                          MLPHP.ACTIVATION: "PReLU",
                          MLPHP.ALPHA: 1.603513,
                          MLPHP.BATCH_SIZE: 15,
                          MLPHP.BETA: 1.88753,
                          MLPHP.LR: 0.143939,
                          MLPHP.N_LAYER: 1,
                          MLPHP.N_UNIT: 2,
                          MLPHP.RHO: 0.868263
                      },
                      E: {
                          MLPHP.ACTIVATION: "PReLU",
                          MLPHP.ALPHA: 0.962996,
                          MLPHP.BATCH_SIZE: 45,
                          MLPHP.BETA: 2.030289,
                          MLPHP.LR: 0.056097,
                          MLPHP.N_LAYER: 0,
                          MLPHP.N_UNIT: 5,
                          MLPHP.RHO: 0.220021
                      },
                      HAN: {
                          HanHP.ALPHA: 0.558703,
                          HanHP.BATCH_SIZE: 15,
                          HanHP.BETA: 1.520712,
                          HanHP.HIDDEN_SIZE: 4,
                          HanHP.LR: 0.109538,
                          HanHP.NUM_HEADS: 19,
                          HanHP.RHO: 1.291954
                      }
                     },
             (B, S, G): {
                         T: {
                             TabNetHP.BATCH_SIZE: 15,
                             TabNetHP.BETA: 0.084007,
                             TabNetHP.GAMMA: 1.684942,
                             TabNetHP.N_A: 5,
                             TabNetHP.N_D: 2,
                             TabNetHP.N_STEPS: 4,
                             TabNetHP.LR: 0.011037
                         },
                         RF: {
                             RandomForestHP.MAX_LEAF_NODES: 25,
                             RandomForestHP.MAX_FEATURES: "sqrt",
                             RandomForestHP.MAX_SAMPLES: 0.994949,
                             RandomForestHP.MIN_SAMPLES_SPLIT: 2,
                             RandomForestHP.N_ESTIMATORS: 1250
                         },
                         XG: {
                             XGBoostHP.ALPHA: 1.883789,
                             XGBoostHP.BETA: 1.757092,
                             XGBoostHP.LR: 0.186442,
                             XGBoostHP.MAX_DEPTH: 1,
                             XGBoostHP.SUBSAMPLE: 1
                         },
                         MLP: {
                             MLPHP.ACTIVATION: "PReLU",
                             MLPHP.ALPHA: 1.409467,
                             MLPHP.BATCH_SIZE: 25,
                             MLPHP.BETA: 0.449466,
                             MLPHP.LR: 0.006111,
                             MLPHP.N_LAYER: 5,
                             MLPHP.N_UNIT: 3,
                             MLPHP.RHO: 0.87755
                         },
                         E: {
                             MLPHP.ACTIVATION: "PReLU",
                             MLPHP.ALPHA: 1.788895,
                             MLPHP.BATCH_SIZE: 45,
                             MLPHP.BETA: 1.786256,
                             MLPHP.LR: 0.05359,
                             MLPHP.N_LAYER: 0,
                             MLPHP.N_UNIT: 5,
                             MLPHP.RHO: 0.862334

                         },
                         HAN: {
                             HanHP.ALPHA: 1.305886,
                             HanHP.BATCH_SIZE: 15,
                             HanHP.BETA: 1.18774,
                             HanHP.HIDDEN_SIZE: 10,
                             HanHP.LR: 0.084286,
                             HanHP.NUM_HEADS: 12,
                             HanHP.RHO: 0.545613
                         }
                        }
             }
