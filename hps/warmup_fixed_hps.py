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
                       TabNetHP.BATCH_SIZE.name: 15,
                       TabNetHP.BETA.name: 0.10656,
                       TabNetHP.GAMMA.name: 1.978489,
                       TabNetHP.N_A.name: 3,
                       TabNetHP.N_D.name: 7,
                       TabNetHP.N_STEPS.name: 5,
                       TabNetHP.LR.name: 0.05207
                    },
                    RF: {
                        RandomForestHP.MAX_LEAF_NODES.name: 25,
                        RandomForestHP.MAX_FEATURES.name: "sqrt",
                        RandomForestHP.MAX_SAMPLES.name: 0.922954,
                        RandomForestHP.MIN_SAMPLES_SPLIT.name: 4,
                        RandomForestHP.N_ESTIMATORS.name: 1250
                    },
                    XG: {
                        XGBoostHP.ALPHA.name: 1.240812,
                        XGBoostHP.BETA.name: 0.968769,
                        XGBoostHP.LR.name: 0.250774,
                        XGBoostHP.MAX_DEPTH.name: 1,
                        XGBoostHP.SUBSAMPLE.name: 1,
                    },
                    MLP: {
                        MLPHP.ACTIVATION.name: "PReLU",
                        MLPHP.ALPHA.name: 0.672994,
                        MLPHP.BATCH_SIZE.name: 55,
                        MLPHP.BETA.name: 0.729828,
                        MLPHP.LR.name: 0.027521,
                        MLPHP.N_LAYER.name: 1,
                        MLPHP.N_UNIT.name: 5,
                        MLPHP.RHO.name: 0.72377
                    },
                    E: {
                        MLPHP.ACTIVATION.name: "PReLU",
                        MLPHP.ALPHA.name: 2.598665,
                        MLPHP.BATCH_SIZE.name: 35,
                        MLPHP.BETA.name: 2.759232,
                        MLPHP.LR.name: 0.462388,
                        MLPHP.N_LAYER.name: 0,
                        MLPHP.N_UNIT.name: 5,
                        MLPHP.RHO.name: 0.73456

                    },
                   },
             (B, S): {
                      T: {
                         TabNetHP.BATCH_SIZE.name: 15,
                         TabNetHP.BETA.name: 0.07302,
                         TabNetHP.GAMMA.name: 1.78896,
                         TabNetHP.N_A.name: 6,
                         TabNetHP.N_D.name: 6,
                         TabNetHP.N_STEPS.name: 5,
                         TabNetHP.LR.name: 0.02654
                      },
                      RF: {
                         RandomForestHP.MAX_LEAF_NODES.name: 25,
                         RandomForestHP.MAX_FEATURES.name: "log2",
                         RandomForestHP.MAX_SAMPLES.name: 0.968991,
                         RandomForestHP.MIN_SAMPLES_SPLIT.name: 3,
                         RandomForestHP.N_ESTIMATORS.name: 2000
                      },
                      XG: {
                          XGBoostHP.ALPHA.name: 1.821231,
                          XGBoostHP.BETA.name: 1.714922,
                          XGBoostHP.LR.name: 0.116989,
                          XGBoostHP.MAX_DEPTH.name: 1,
                          XGBoostHP.SUBSAMPLE.name: 1

                      },
                      MLP: {
                          MLPHP.ACTIVATION.name: "PReLU",
                          MLPHP.ALPHA.name: 1.603513,
                          MLPHP.BATCH_SIZE.name: 15,
                          MLPHP.BETA.name: 1.88753,
                          MLPHP.LR.name: 0.143939,
                          MLPHP.N_LAYER.name: 1,
                          MLPHP.N_UNIT.name: 2,
                          MLPHP.RHO.name: 0.868263
                      },
                      E: {
                          MLPHP.ACTIVATION.name: "PReLU",
                          MLPHP.ALPHA.name: 0.962996,
                          MLPHP.BATCH_SIZE.name: 45,
                          MLPHP.BETA.name: 2.030289,
                          MLPHP.LR.name: 0.056097,
                          MLPHP.N_LAYER.name: 0,
                          MLPHP.N_UNIT.name: 5,
                          MLPHP.RHO.name: 0.220021
                      },
                      HAN: {
                          HanHP.ALPHA.name: 0.558703,
                          HanHP.BATCH_SIZE.name: 15,
                          HanHP.BETA.name: 1.520712,
                          HanHP.HIDDEN_SIZE.name: 4,
                          HanHP.LR.name: 0.109538,
                          HanHP.NUM_HEADS.name: 19,
                          HanHP.RHO.name: 1.291954
                      }
                     },
             (B, S, G): {
                         T: {
                             TabNetHP.BATCH_SIZE.name: 15,
                             TabNetHP.BETA.name: 0.084007,
                             TabNetHP.GAMMA.name: 1.684942,
                             TabNetHP.N_A.name: 5,
                             TabNetHP.N_D.name: 2,
                             TabNetHP.N_STEPS.name: 4,
                             TabNetHP.LR.name: 0.011037
                         },
                         RF: {
                             RandomForestHP.MAX_LEAF_NODES.name: 25,
                             RandomForestHP.MAX_FEATURES.name: "sqrt",
                             RandomForestHP.MAX_SAMPLES.name: 0.994949,
                             RandomForestHP.MIN_SAMPLES_SPLIT.name: 2,
                             RandomForestHP.N_ESTIMATORS.name: 1250
                         },
                         XG: {
                             XGBoostHP.ALPHA.name: 1.883789,
                             XGBoostHP.BETA.name: 1.757092,
                             XGBoostHP.LR.name: 0.186442,
                             XGBoostHP.MAX_DEPTH.name: 1,
                             XGBoostHP.SUBSAMPLE.name: 1
                         },
                         MLP: {
                             MLPHP.ACTIVATION.name: "PReLU",
                             MLPHP.ALPHA.name: 1.409467,
                             MLPHP.BATCH_SIZE.name: 25,
                             MLPHP.BETA.name: 0.449466,
                             MLPHP.LR.name: 0.006111,
                             MLPHP.N_LAYER.name: 5,
                             MLPHP.N_UNIT.name: 3,
                             MLPHP.RHO.name: 0.87755
                         },
                         E: {
                             MLPHP.ACTIVATION.name: "PReLU",
                             MLPHP.ALPHA.name: 1.788895,
                             MLPHP.BATCH_SIZE.name: 45,
                             MLPHP.BETA.name: 1.786256,
                             MLPHP.LR.name: 0.05359,
                             MLPHP.N_LAYER.name: 0,
                             MLPHP.N_UNIT.name: 5,
                             MLPHP.RHO.name: 0.862334

                         },
                         HAN: {
                             HanHP.ALPHA.name: 1.305886,
                             HanHP.BATCH_SIZE.name: 15,
                             HanHP.BETA.name: 1.18774,
                             HanHP.HIDDEN_SIZE.name: 10,
                             HanHP.LR.name: 0.084286,
                             HanHP.NUM_HEADS.name: 12,
                             HanHP.RHO.name: 0.545613
                         }
                        }
             }
