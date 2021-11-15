"""
Filename: breast_cancer_test.py

Author: Nicolas Raymond

Description: This file is used to test the evaluator class
             with all models, using breast cancer dataset.

Date of last modification: 2021/11/11
"""
import sys
from copy import deepcopy
from os.path import dirname, realpath
from sklearn.datasets import load_breast_cancer

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from hps.sanity_check_hps import TAB_HPS, RF_HPS, XGBOOST_HPS, MLP_HPS
    from src.data.extraction.constants import SEED, PARTICIPANT
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.sampling import RandomStratifiedSampler, push_valid_to_train
    from src.models.mlp import PetaleBinaryMLPC
    from src.models.random_forest import PetaleBinaryRFC
    from src.models.tabnet import PetaleBinaryTNC
    from src.models.xgboost_ import PetaleBinaryXGBC
    from src.utils.score_metrics import BinaryAccuracy, BinaryBalancedAccuracy,\
        BalancedAccuracyEntropyRatio, Sensitivity, Specificity, Reduction
    from src.training.evaluation import Evaluator

    # Data loading
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.data
    cont_col = df.columns
    target = 'target'
    df[target] = dataset.target
    df[PARTICIPANT] = list(range(df.shape[0]))

    # Dataset creation
    dataset = PetaleDataset(df, target, cont_col)

    # Masks creation
    sampler = RandomStratifiedSampler(dataset=dataset,
                                      n_out_split=3,
                                      n_in_split=5,
                                      random_state=SEED,
                                      alpha=10)
    masks = sampler()

    # Creation of another mask without valid
    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [BinaryAccuracy(), BinaryBalancedAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(), Specificity(),
                          BalancedAccuracyEntropyRatio(Reduction.GEO_MEAN)]

    """
    Evaluator validation with TabNet
    """
    # Creation of dataset
    dataset = PetaleDataset(df, target, cont_col)

    # Saving of fixed params for TabNet
    def update_fixed_params(subset):
        return {'cat_idx': [], 'cat_sizes': [],
                'cat_emb_sizes': [], 'max_epochs': 250,
                'patience': 50}

    fixed_params = update_fixed_params(dataset)

    # Creation of the evaluator
    evaluator = Evaluator(model_constructor=PetaleBinaryTNC,
                          dataset=dataset,
                          masks=masks,
                          hps=TAB_HPS,
                          n_trials=250,
                          fixed_params=fixed_params,
                          fixed_params_update_function=update_fixed_params,
                          evaluation_metrics=evaluation_metrics,
                          evaluation_name='TabNet_test',
                          save_hps_importance=True,
                          save_optimization_history=True,
                          seed=SEED)

    # Evaluation
    evaluator.evaluate()

    """
    Evaluator validation with RF
    """
    evaluator = Evaluator(model_constructor=PetaleBinaryRFC,
                          dataset=dataset,
                          masks=masks_without_val,
                          hps=RF_HPS,
                          n_trials=250,
                          evaluation_metrics=evaluation_metrics,
                          evaluation_name='RF_test',
                          save_hps_importance=True,
                          save_optimization_history=True,
                          seed=SEED)

    evaluator.evaluate()

    """
    Evaluator validation with XGBoost
    """
    evaluator = Evaluator(model_constructor=PetaleBinaryXGBC,
                          dataset=dataset,
                          masks=masks_without_val,
                          hps=XGBOOST_HPS,
                          n_trials=250,
                          evaluation_metrics=evaluation_metrics,
                          evaluation_name='XGBoost_test',
                          save_hps_importance=True,
                          save_optimization_history=True,
                          seed=SEED)

    evaluator.evaluate()

    """
    Evaluator validation with MLP
    """
    dataset_mlp = PetaleDataset(df, target, cont_col, to_tensor=True)

    # Saving of fixed_params for MLP
    def update_fixed_params(subset):
        return {'max_epochs': 250,
                'patience': 50,
                'num_cont_col': len(subset.cont_cols),
                'cat_idx': subset.cat_idx,
                'cat_sizes': subset.cat_sizes,
                'cat_emb_sizes': subset.cat_sizes}

    fixed_params = update_fixed_params(dataset_mlp)

    evaluator = Evaluator(model_constructor=PetaleBinaryMLPC,
                          dataset=dataset_mlp, masks=masks,
                          hps=MLP_HPS,
                          n_trials=250,
                          evaluation_metrics=evaluation_metrics,
                          fixed_params=fixed_params,
                          evaluation_name='MLP_test',
                          fixed_params_update_function=update_fixed_params,
                          save_optimization_history=True,
                          seed=SEED)

    evaluator.evaluate()
