"""
This file consists of all the experiments made on the l1 dataset
"""
from os.path import dirname, realpath, join
import sys
import argparse
import time


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 full_experiment.py',
                                     description="Runs all the experiments associated to the l1 dataset")

    # Nb inner split and nb outer split selection
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=5,
                        help='Number of outer splits during the models evaluations')
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=5,
                        help='Number of inner splits during the models evaluations')

    # Features selection
    parser.add_argument('-base', '--baselines', default=False, action='store_true',
                        help='True if we want to add baselines features into dataset')
    parser.add_argument('-comp', '--complication', type=str, default='bone',
                        choices=['bone', 'cardio', 'neuro', 'all'], help='Choice of health complication to predict')
    parser.add_argument('-gen', '--genes', type=str, default=None, choices=[None, 'significant', 'all'],
                        help="Selection of genes to incorporate into the dataset")

    # Models selection
    parser.add_argument('-han', '--han', default=False, action='store_true',
                        help='True if we want to run heterogeneous graph attention network experiments')
    parser.add_argument('-logit', '--logistic_regression', default=False, action='store_true',
                        help='True if we want to run logistic regression experiments')
    parser.add_argument('-mlp', '--mlp', default=False, action='store_true',
                        help='True if we want to run mlp experiments')
    parser.add_argument('-rf', '--random_forest', default=False, action='store_true',
                        help='True if we want to run random forest experiments')
    parser.add_argument('-xg', '--xg_boost', default=False, action='store_true',
                        help='True if we want to run xgboost experiments')
    parser.add_argument('-tab', '--tabnet', default=False, action='store_true',
                        help='True if we want to run TabNet experiments')

    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=SEED, help='Seed to use during model evaluations')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from hps.l1_hps import TAB_HPS, RF_HPS, XGBOOST_HPS, HAN_HPS, MLP_HPS, LOGIT_HPS
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset, PetaleStaticGNNDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.sampling import get_learning_one_data, extract_masks
    from src.models.han import PetaleBinaryHANC
    from src.models.mlp import PetaleBinaryMLPC
    from src.models.tabnet import PetaleBinaryTNC
    from src.models.random_forest import PetaleBinaryRFC
    from src.models.xgboost_ import PetaleBinaryXGBC
    from src.training.evaluation import Evaluator
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.score_metrics import BinaryAccuracy, BinaryBalancedAccuracy, \
        BalancedAccuracyEntropyRatio, Sensitivity, Specificity, Reduction

    # Arguments parsing
    args = argument_parser()

    # Extraction of complication choice
    complication = args.complication
    if complication == 'bone':
        complication = BONE_COMPLICATIONS
    elif complication == 'cardio':
        complication = CARDIOMETABOLIC_COMPLICATIONS
    elif complication == 'neuro':
        complication = NEUROCOGNITIVE_COMPLICATIONS
    else:
        complication = COMPLICATIONS

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We extract data
    df, cont_cols, cat_cols = get_learning_one_data(manager, baselines=args.baselines, genes=args.genes,
                                                    complications=[complication])
    # Extraction of masks
    masks = extract_masks(join(Paths.MASKS, "l1_masks.json"), k=args.nb_outer_splits, l=args.nb_inner_splits)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [BinaryAccuracy(), BinaryBalancedAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(), Specificity(),
                          BalancedAccuracyEntropyRatio(Reduction.GEO_MEAN)]

    # Initialization of feature selector
    feature_selector = FeatureSelector(0.95)

    # We start a timer for the whole experiment
    first_start = time.time()

    """
    TabNet experiment
    """
    if args.tabnet:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, complication, cont_cols, cat_cols)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'cat_idx': dts.cat_idx, 'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes, 'max_epochs': 250,
                    'patience': 50}

        # Saving of original fixed params for TabNet
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleBinaryTNC, dataset=dataset,
                              evaluation_name=f"L1_TabNet_{args.complication}_{args.genes}",
                              masks=masks, hps=TAB_HPS, n_trials=100, fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              feature_selector=feature_selector,
                              evaluation_metrics=evaluation_metrics,
                              save_hps_importance=True, save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for TabNet (minutes): ", round((time.time() - start) / 60, 2))

    """
    Random Forest experiment
    """
    if args.random_forest:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, complication, cont_cols, cat_cols)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleBinaryRFC, dataset=dataset, masks=masks,
                              evaluation_name=f"L1_RandomForest_{args.complication}_{args.genes}",
                              hps=RF_HPS, n_trials=100, evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector, save_hps_importance=True,
                              save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for Random Forest (minutes): ", round((time.time() - start) / 60, 2))

    """
    XGBoost experiment
    """
    if args.xg_boost:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, complication, cont_cols, cat_cols)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleBinaryXGBC, dataset=dataset, masks=masks,
                              evaluation_name=f"L1_XGBoost_{args.complication}_{args.genes}",
                              hps=XGBOOST_HPS, n_trials=100, evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector, save_hps_importance=True,
                              save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for XGBoost (minutes): ", round((time.time() - start) / 60, 2))

    """
    MLP experiment
    """
    if args.mlp:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, complication, cont_cols, cat_cols, to_tensor=True)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': 250, 'patience': 50, 'num_cont_col': len(dts.cont_cols),
                    'cat_idx': dts.cat_idx, 'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of fixed_params for MLP
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleBinaryMLPC, dataset=dataset, masks=masks,
                              evaluation_name=f"L1_MLP_{args.complication}_{args.genes}",
                              hps=MLP_HPS, n_trials=100, evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector, fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True, save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for MLP (minutes): ", round((time.time() - start) / 60, 2))

    """
    HAN experiment
    """
    if args.han:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleStaticGNNDataset(df, complication, cont_cols, cat_cols)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'meta_paths': dts.get_metapaths(), 'in_size': len(dts.cont_cols),
                    'max_epochs': 250, 'patience': 15}


        # Saving of original fixed params for HAN
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleBinaryHANC, dataset=dataset, masks=masks,
                              evaluation_name=f"L1_HAN_{args.complication}_{args.genes}",
                              hps=HAN_HPS, n_trials=50, evaluation_metrics=evaluation_metrics,
                              fixed_params=fixed_params, fixed_params_update_function=update_fixed_params,
                              feature_selector=feature_selector, save_hps_importance=True,
                              save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for HAN (minutes): ", round((time.time() - start) / 60, 2))

    """
    Logistic regression experiment
    """
    if args.logistic_regression:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, complication, cont_cols, cat_cols, to_tensor=True)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': 1500, 'patience': 200, 'num_cont_col': len(dts.cont_cols),
                    'cat_idx': dts.cat_idx, 'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}


        # Saving of fixed_params for MLP
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleBinaryMLPC, dataset=dataset, masks=masks,
                              evaluation_name=f"L1_Logit_{args.complication}_{args.genes}",
                              hps=LOGIT_HPS, n_trials=100, evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector, fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True, save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for Logistic Regression (minutes): ", round((time.time() - start) / 60, 2))

    print("Overall time (minutes): ", round((time.time() - first_start) / 60, 2))