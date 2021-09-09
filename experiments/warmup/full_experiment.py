"""
This file consists of all the experiments made on the warmup dataset
"""
from os.path import dirname, realpath
from copy import deepcopy
import sys
import argparse
import time


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python full_experiment.py',
                                     description="Runs all the experiments associated to the warmup dataset")

    # Nb inner split and nb outer split selection
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=5,
                        help='Number of outer splits during the models evaluations')
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=10,
                        help='Number of inner splits during the models evaluations')

    # Features selection
    parser.add_argument('-gen', '--genes', default=False, action='store_true',
                        help='True if we want to include genes if features')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='True if we want to apply automatic feature selection')
    parser.add_argument('-s', '--sex', default=False, action='store_true',
                        help='True if we want to include sex in features')

    # Models selection
    parser.add_argument('-han', '--han', default=False, action='store_true',
                        help='True if we want to run heterogeneous graph attention network experiments')
    parser.add_argument('-lin', '--linear_regression', default=False, action='store_true',
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
    from hps.warmup_hps import TAB_HPS, RF_HPS, HAN_HPS, MLP_HPS, ENET_HPS, XGBOOST_HPS
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset, PetaleStaticGNNDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.sampling import get_warmup_data, extract_masks, push_valid_to_train
    from src.models.han import PetaleHANR
    from src.models.mlp import PetaleMLPR
    from src.models.tabnet import PetaleTNR
    from src.models.random_forest import PetaleRFR
    from src.models.xgboost_ import PetaleXGBR
    from src.training.evaluation import Evaluator
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError, SquaredError

    # Arguments parsing
    args = argument_parser()

    # Initialization of DataManager and sampler
    manager = PetaleDataManager("rayn2402")

    # We extract needed data
    df, target, cont_cols, cat_cols = get_warmup_data(manager, genes=args.genes, sex=args.sex)

    # Extraction of masks
    masks = extract_masks(Paths.WARMUP_MASK, k=args.nb_outer_splits, l=args.nb_inner_splits)
    gnn_masks = extract_masks(Paths.WARMUP_MASK, k=args.nb_outer_splits, l=min(args.nb_inner_splits, 2))
    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [AbsoluteError(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Initialization of feature selector
    if args.feature_selection:
        feature_selector = FeatureSelector(0.95)
    else:
        feature_selector = None

    # We save the string that will help identify evaluations
    eval_id = ""
    if args.genes:
        eval_id = f"{eval_id}_genes"
    if args.sex:
        eval_id = f"{eval_id}_sex"

    # We start a timer for the whole experiment
    first_start = time.time()

    """
    TabNet experiment
    """
    if args.tabnet:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            if len(dts.cat_idx) != 0:
                return {'cat_idx': dts.cat_idx, 'cat_sizes': dts.cat_sizes,
                        'cat_emb_sizes': dts.cat_sizes, 'max_epochs': 250,
                        'patience': 50}
            else:
                return {'cat_idx': [], 'cat_sizes': [],
                        'cat_emb_sizes': [], 'max_epochs': 250,
                        'patience': 50}

        # Saving of original fixed params for TabNet
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleTNR, dataset=dataset,
                              evaluation_name=f"TabNet_warmup_{eval_id}",
                              masks=masks, hps=TAB_HPS, n_trials=200, fixed_params=fixed_params,
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
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleRFR, dataset=dataset, masks=masks_without_val,
                              evaluation_name=f"RandomForest_warmup_{eval_id}",
                              hps=RF_HPS, n_trials=200, evaluation_metrics=evaluation_metrics,
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
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleXGBR, dataset=dataset, masks=masks_without_val,
                              evaluation_name=f"XGBoost_warmup_{eval_id}",
                              hps=XGBOOST_HPS, n_trials=200, evaluation_metrics=evaluation_metrics,
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
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': 250, 'patience': 50, 'num_cont_col': len(dts.cont_cols),
                    'cat_idx': dts.cat_idx, 'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of fixed_params for MLP
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR, dataset=dataset, masks=masks,
                              evaluation_name=f"MLP_warmup_{eval_id}",
                              hps=MLP_HPS, n_trials=200, evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector, fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True, save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for MLP (minutes): ", round((time.time() - start) / 60, 2))

    """
    Linear regression experiment
    """
    if args.linear_regression:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': 50, 'patience': 25, 'num_cont_col': len(dts.cont_cols),
                    'cat_idx': dts.cat_idx, 'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}


        # Saving of fixed_params for MLP
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        m = masks_without_val if not args.genes else masks
        evaluator = Evaluator(model_constructor=PetaleMLPR, dataset=dataset, masks=m,
                              evaluation_name=f"linear_reg_warmup_{eval_id}",
                              hps=ENET_HPS, n_trials=200, evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector, fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True, save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for Logistic Regression (minutes): ", round((time.time() - start) / 60, 2))

    """
    HAN experiment
    """
    if args.han and (args.genes or args.sex):

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleStaticGNNDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'meta_paths': dts.get_metapaths(), 'in_size': len(dts.cont_cols),
                    'max_epochs': 250, 'patience': 15}


        # Saving of original fixed params for HAN
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleHANR, dataset=dataset, masks=gnn_masks,
                              evaluation_name=f"HAN_warmup_{eval_id}",
                              hps=HAN_HPS, n_trials=100, evaluation_metrics=evaluation_metrics,
                              fixed_params=fixed_params, fixed_params_update_function=update_fixed_params,
                              feature_selector=feature_selector, save_hps_importance=True,
                              save_optimization_history=True)

        # Evaluation
        evaluator.nested_cross_valid()

        print("Time Taken for HAN (minutes): ", round((time.time() - start) / 60, 2))

    print("Overall time (minutes): ", round((time.time() - first_start) / 60, 2))