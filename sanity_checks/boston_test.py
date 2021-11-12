"""
Filename: boston_test.py

Author: Nicolas Raymond

Description: This file is used to test the evaluator class
             with all models, using boston housing-prices dataset.

Date of last modification: 2021/11/11
"""

import sys
from copy import deepcopy
from os.path import dirname, realpath
from pandas import DataFrame
from sklearn.datasets import load_boston

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from hps.sanity_check_hps import TAB_HPS, RF_HPS, XGBOOST_HPS, MLP_HPS, HAN_HPS
    from src.data.extraction.constants import SEED, PARTICIPANT
    from src.data.processing.datasets import PetaleDataset, PetaleStaticGNNDataset
    from src.data.processing.sampling import RandomStratifiedSampler, push_valid_to_train
    from src.models.han import PetaleHANR
    from src.models.mlp import PetaleMLPR
    from src.models.random_forest import PetaleRFR
    from src.models.tabnet import PetaleTNR
    from src.models.xgboost_ import PetaleXGBR
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError, SquaredError
    from src.training.evaluation import Evaluator

    # Data loading
    dataset = load_boston()
    data, prices, column_names = dataset.data, dataset.target, list(dataset.feature_names)
    cat_col = ['RAD']
    cont_col = [c for c in column_names if c not in ['RAD', 'CHAS']]  # We remove CHAS
    df = DataFrame(data=data, columns=column_names)
    df = df[cont_col+cat_col]
    target = 'target'
    df[target] = prices
    print(f"Min price : {df[target].min()}, Max price : {df[target].max()}, Mean price : {df[target].mean()}")
    df[PARTICIPANT] = list(range(df.shape[0]))

    # Dataset creation
    dataset = PetaleDataset(df, target, cont_col, cat_col, classification=False)

    # Masks creation
    sampler = RandomStratifiedSampler(dataset=dataset,
                                      n_out_split=3,
                                      n_in_split=5,
                                      random_state=SEED,
                                      alpha=15)
    masks = sampler()

    # Creation of another mask without valid
    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [SquaredError(), AbsoluteError(), Pearson(), RootMeanSquaredError()]

    """
    Test with TabNet
    """
    # Creation of dataset
    dataset = PetaleDataset(df, target, cont_col, cat_col, classification=False)

    # Saving of fixed params for TabNet
    def update_fixed_params(subset):
        return {'cat_idx': subset.cat_idx,
                'cat_sizes': subset.cat_sizes,
                'cat_emb_sizes': subset.cat_sizes,
                'max_epochs': 250,
                'patience': 50}

    fixed_params = update_fixed_params(dataset)

    # Creation of the evaluator
    evaluator = Evaluator(model_constructor=PetaleTNR,
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
    Test with RF
    """
    evaluator = Evaluator(model_constructor=PetaleRFR,
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
    Test with XGBoost
    """
    evaluator = Evaluator(model_constructor=PetaleXGBR,
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
    Test with MLP
    """
    dataset_mlp = PetaleDataset(df, target, cont_col, cat_col, classification=False, to_tensor=True)

    # Saving of fixed_params for MLP
    def update_fixed_params(subset):
        return {'max_epochs': 250,
                'patience': 50,
                'num_cont_col': len(subset.cont_cols),
                'cat_idx': subset.cat_idx,
                'cat_sizes': subset.cat_sizes,
                'cat_emb_sizes': subset.cat_sizes}

    fixed_params = update_fixed_params(dataset_mlp)

    evaluator = Evaluator(model_constructor=PetaleMLPR,
                          dataset=dataset_mlp,
                          masks=masks,
                          hps=MLP_HPS,
                          n_trials=250,
                          evaluation_metrics=evaluation_metrics,
                          fixed_params=fixed_params,
                          evaluation_name='MLP_test',
                          fixed_params_update_function=update_fixed_params,
                          save_optimization_history=True,
                          seed=SEED)

    evaluator.evaluate()

    """
    Test with HAN
    """
    dataset_gnn = PetaleStaticGNNDataset(df, target, cont_col, cat_col, classification=False)

    # Saving of fixed params for TabNet
    def update_fixed_params(subset):
        return {'meta_paths': subset.get_metapaths(),
                'num_cont_col': len(subset.cont_cols),
                'cat_idx': subset.cat_idx,
                'cat_sizes': subset.cat_sizes,
                'cat_emb_sizes': subset.cat_sizes,
                'max_epochs': 250,
                'patience': 15}

    fixed_params = update_fixed_params(dataset_gnn)

    evaluator = Evaluator(model_constructor=PetaleHANR,
                          dataset=dataset_gnn,
                          masks=masks,
                          hps=HAN_HPS,
                          n_trials=250,
                          evaluation_metrics=evaluation_metrics,
                          fixed_params=fixed_params,
                          evaluation_name='HAN_test',
                          fixed_params_update_function=update_fixed_params,
                          save_optimization_history=True,
                          seed=SEED)

    evaluator.evaluate()
