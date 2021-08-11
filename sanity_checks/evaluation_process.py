"""
This file is used to validate the NNEvaluator and RFEvaluator classes
"""


import sys
from os.path import join, realpath, dirname


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(realpath(__file__))))
    from sanity_checks.hps import TAB_HPS, RF_HPS, XGBOOST_HPS, HAN_HPS, MLP_HPS
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset, PetaleStaticGNNDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.sampling import get_learning_one_data, extract_masks, ALL
    from src.models.han import PetaleBinaryHANC
    from src.models.mlp import PetaleBinaryMLPC
    from src.models.tabnet import PetaleBinaryTNC
    from src.models.random_forest import PetaleBinaryRFC
    from src.models.xgboost_ import PetaleBinaryXGBC
    from src.training.evaluation import Evaluator
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.score_metrics import BinaryAccuracy, BinaryBalancedAccuracy,\
        BalancedAccuracyEntropyRatio, Sensitivity, Specificity, Reduction

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=True, genes=ALL,
                                                    complications=[NEUROCOGNITIVE_COMPLICATIONS])
    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "l1_masks.json"), k=2, l=5)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [BinaryAccuracy(), BinaryBalancedAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(), Specificity(),
                          BalancedAccuracyEntropyRatio(Reduction.GEO_MEAN)]

    feature_selector = FeatureSelector(0.90)

    """
    Evaluator validation with TabNet
    """
    # Creation of dataset
    dataset = PetaleDataset(df, NEUROCOGNITIVE_COMPLICATIONS, cont_cols, cat_cols)

    # Saving of fixed params for TabNet
    fixed_params = {'cat_idx': dataset.cat_idx, 'cat_sizes': dataset.cat_sizes,
                    'cat_emb_sizes': dataset.cat_sizes, 'max_epochs': 250,
                    'patience': 50}

    def update_fixed_params(subset):
        return {'cat_idx': subset.cat_idx, 'cat_sizes': subset.cat_sizes,
                'cat_emb_sizes': subset.cat_sizes, 'max_epochs': 250,
                'patience': 50}


    # Creation of the evaluator
    evaluator = Evaluator(model_constructor=PetaleBinaryTNC, dataset=dataset,
                          masks=masks, hps=TAB_HPS, n_trials=5, fixed_params=fixed_params,
                          fixed_params_update_function=update_fixed_params,
                          feature_selector=feature_selector,
                          evaluation_metrics=evaluation_metrics,
                          save_hps_importance=True, save_optimization_history=True)

    # Evaluation
    evaluator.nested_cross_valid()

    """
    Evaluator validation with RF
    """
    evaluator = Evaluator(model_constructor=PetaleBinaryRFC, dataset=dataset, masks=masks,
                          hps=RF_HPS, n_trials=10, evaluation_metrics=evaluation_metrics,
                          feature_selector=feature_selector, save_hps_importance=True,
                          save_optimization_history=True)

    evaluator.nested_cross_valid()

    """
    Evaluator validation with XGBoost
    """
    evaluator = Evaluator(model_constructor=PetaleBinaryXGBC, dataset=dataset, masks=masks,
                          hps=XGBOOST_HPS, n_trials=10, evaluation_metrics=evaluation_metrics,
                          feature_selector=feature_selector, save_hps_importance=True,
                          save_optimization_history=True)

    evaluator.nested_cross_valid()

    """
    Evaluator validation with HAN
    """
    dataset_gnn = PetaleStaticGNNDataset(df, NEUROCOGNITIVE_COMPLICATIONS, cont_cols, cat_cols)

    # Saving of fixed params for TabNet
    fixed_params = {'meta_paths': dataset_gnn.get_metapaths(), 'in_size': len(dataset_gnn.cont_cols),
                    'max_epochs': 250, 'patience': 15}

    def update_fixed_params(subset):
        return {'meta_paths': subset.get_metapaths(), 'in_size': len(subset.cont_cols),
                'max_epochs': 250, 'patience': 15}

    evaluator = Evaluator(model_constructor=PetaleBinaryHANC, dataset=dataset_gnn, masks=masks,
                          hps=HAN_HPS, n_trials=10, evaluation_metrics=evaluation_metrics,
                          fixed_params=fixed_params, fixed_params_update_function=update_fixed_params,
                          feature_selector=feature_selector, save_optimization_history=True)

    evaluator.nested_cross_valid()

    """
    Evaluator validation with MLP
    """
    dataset_mlp = PetaleDataset(df, NEUROCOGNITIVE_COMPLICATIONS, cont_cols, cat_cols, to_tensor=True)

    # Saving of fixed_params for MLP
    fixed_params = {'max_epochs': 250, 'patience': 50, 'num_cont_col': len(dataset_mlp.cont_cols),
                    'cat_idx': dataset_mlp.cat_idx, 'cat_sizes': dataset_mlp.cat_sizes,
                    'cat_emb_sizes': dataset_mlp.cat_sizes}

    def update_fixed_params(subset):
        return {'max_epochs': 250, 'patience': 50, 'num_cont_col': len(subset.cont_cols),
                'cat_idx': subset.cat_idx, 'cat_sizes': subset.cat_sizes,
                'cat_emb_sizes': subset.cat_sizes}


    evaluator = Evaluator(model_constructor=PetaleBinaryMLPC, dataset=dataset_mlp, masks=masks,
                          hps=MLP_HPS, n_trials=10, evaluation_metrics=evaluation_metrics, fixed_params=fixed_params,
                          fixed_params_update_function=update_fixed_params, save_optimization_history=True)

    evaluator.nested_cross_valid()
