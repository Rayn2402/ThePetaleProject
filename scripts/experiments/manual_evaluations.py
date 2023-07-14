"""
Filename: manual_evaluations.py

Author: Nicolas Raymond

Description: Script used to run VO2 peak experiments using
             manually selected hyperparameters.

Date of last modification: 2022/07/27
"""

import sys
import time

from argparse import ArgumentParser
from copy import deepcopy
from os.path import dirname, realpath


def retrieve_arguments():
    """
    Creates a parser for VO2 peak prediction experiments
    """
    # Create a parser
    parser = ArgumentParser(usage='\n python [experiment file].py',
                            description="Runs the experiments associated to the VO2 dataset.")

    # Nb of inner splits and nb of outer splits
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=10,
                        help='Number of outer splits used during the models evaluations.')
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=10,
                        help='Number of inner splits used during the models evaluations.')

    # Data source
    parser.add_argument('-from_csv', '--from_csv', default=False, action='store_true',
                        help='If true, extract the data from the csv file instead of the database.')

    # Feature selection
    parser.add_argument('-r_w', '--remove_walk_variables', default=False, action='store_true',
                        help='If true, removes the six-minute walk test variables from the data.')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='If true, applies automatic feature selection')

    # Models selection
    parser.add_argument('-enet', '--enet', default=False, action='store_true',
                        help='If true, runs enet experiment')
    parser.add_argument('-mlp', '--mlp', default=False, action='store_true',
                        help='If true, runs mlp experiment')
    parser.add_argument('-rf', '--random_forest', default=False, action='store_true',
                        help='If true, runs random forest experiment')
    parser.add_argument('-xg', '--xg_boost', default=False, action='store_true',
                        help='If true, runs xgboost experiment')
    parser.add_argument('-gat', '--gat', default=False, action='store_true',
                        help='If true, runs Graph Attention Network experiment')
    parser.add_argument('-gcn', '--gcn', default=False, action='store_true',
                        help='If true, runs Graph Convolutional Network experiment')

    # Training parameters
    parser.add_argument('-epochs', '--epochs', type=int, default=100,
                        help='Maximal number of epochs during training')
    parser.add_argument('-patience', '--patience', type=int, default=10,
                        help='Number of epochs allowed without improvement (for early stopping)')

    # Graph construction parameters
    parser.add_argument('-w_sim', '--weighted_similarity', default=False, action='store_true',
                        help='If true, calculates patients similarities using weighted metrics')
    parser.add_argument('-cond_col', '--conditional_column', default=False, action='store_true',
                        help='If true, uses the sex as a conditional column in graph construction')
    parser.add_argument('-deg', '--degree', nargs='*', type=str, default=[7],
                        help="Maximum number of in-degrees for each node in the graph")

    # Activation of sharpness-aware minimization
    parser.add_argument('-rho', '--rho', type=float, default=0,
                        help='Rho parameter of Sharpness-Aware Minimization (SAM) Optimizer.'
                             'If >0, SAM is enabled')

    # Usage of predictions from another experiment
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path leading to predictions of another model')

    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=1010710,
                        help='Seed used during model evaluations')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print(f"{arg}: {getattr(arguments, arg)}")
    print("\n")

    return arguments


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append((dirname(dirname(dirname(realpath(__file__))))))
    from hps import manually_selected_hps as ms_hps
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.sampling import extract_masks, get_VO2_data, push_valid_to_train
    from src.models.gcn import PetaleGCNR, GCNHP
    from src.models.gas import PetaleGASR, GASHP
    from src.models.gat import PetaleGATR, GATHP
    from src.models.mlp import PetaleMLPR, MLPHP
    from src.models.random_forest import PetaleRFR
    from src.models.xgboost_ import PetaleXGBR
    from src.evaluation.evaluation import Evaluator
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.metrics import AbsoluteError, Pearson, RootMeanSquaredError, SpearmanR, SquaredError

    # Arguments parsing
    args = retrieve_arguments()

    # Initialization of a data manager
    manager = PetaleDataManager() if not args.from_csv else None

    # We extract the data needed
    df, target, cont_cols, cat_cols = get_VO2_data(manager)

    # We filter baselines variables if needed
    if args.remove_walk_variables:
        df.drop([TDM6_HR_END, TDM6_DIST], axis=1, inplace=True)
        cont_cols = [c for c in cont_cols if c not in [TDM6_HR_END, TDM6_DIST]]

    # Extraction of masks
    masks = extract_masks(Paths.VO2_MASK, k=args.nb_outer_splits, l=args.nb_inner_splits)

    # Creation of masks for tree-based models
    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of list containing the evaluation metrics
    evaluation_metrics = [AbsoluteError(), SpearmanR(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Initialization of a feature selector
    if args.feature_selection:
        feature_selector = FeatureSelector(threshold=[0.01],
                                           cumulative_imp=[False],
                                           seed=args.seed)
    else:
        feature_selector = None

    # We save the string that will help identify evaluations
    eval_id = "vo2_manual"
    if args.remove_walk_variables:
        eval_id += "_nw"
    if args.rho > 0:
        eval_id += "_sam"

    # We start a timer for the whole experiment
    first_start = time.time()

    """
    Random Forest experiment
    """
    if args.random_forest:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleRFR,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"RF_{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=ms_hps.RF_HPS,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for Random Forest (min): {(time.time() - start)/60:.2f}")

    """
    XGBoost experiment
    """
    if args.xg_boost:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleXGBR,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"XGBoost_{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=ms_hps.XGBOOST_HPS,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for XGBoost (min): {(time.time() - start)/60:.2f}")

    """
    MLP experiment
    """
    if args.mlp:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Update of the hyperparameters
        ms_hps.MLP_HPS[MLPHP.RHO.name] = args.rho
        cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
        ms_hps.MLP_HPS[MLPHP.N_UNIT.name] = int((len(cont_cols) + cat_sizes_sum)/2)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    **ms_hps.MLP_HPS}

        # Saving of the fixed params of MLP
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"MLP_{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for MLP (min): {(time.time() - start)/60:.2f}")

    """
    ENET experiment
    """
    if args.enet:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Update of the hyperparameters
        ms_hps.ENET_HPS[MLPHP.RHO.name] = args.rho

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    **ms_hps.ENET_HPS}

        # Saving of the fixed params of ENET
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"enet_{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for enet (min): {(time.time() - start)/60:.2f}")

    """
    GAS experiment
    """
    if args.gas and (args.path is not None):

        # Start timer
        start = time.time()

        # Update of the hyperparameters
        ms_hps.GASHPS[GASHP.RHO.name] = args.rho

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'num_cont_col': len(dts.cont_idx),
                    'previous_pred_idx': len(dts.cont_idx) - 1,
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'max_epochs': args.epochs,
                    'patience': args.patience,
                    **ms_hps.GASHPS}

        # Saving of the fixed params of GAT
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleGASR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"GAS_{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for GAS (min): {(time.time() - start)/60:.2f}")

    """
    GAT experiment
    """
    if args.gat:

        # Start timer
        start = time.time()

        # Update of the hyperparameters
        ms_hps.GATHPS[GATHP.RHO.name] = args.rho

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'max_epochs': args.epochs,
                    'patience': args.patience,
                    **ms_hps.GATHPS}

        for nb_neighbor in args.degree:

            # We change the type from str to int
            nb_neigh = int(nb_neighbor)

            # We set the conditional column
            cond_cat_col = SEX if args.conditional_column else None

            # We set the distance computations options
            GAT_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

            for prefix, w_sim in GAT_options:

                # Creation of the dataset
                dataset = PetaleKGNNDataset(df, target, k=nb_neigh,
                                            weighted_similarity=w_sim,
                                            cont_cols=cont_cols, cat_cols=cat_cols,
                                            conditional_cat_col=cond_cat_col,
                                            classification=False)

                # Saving of the fixed params of GAT
                fixed_params = update_fixed_params(dataset)

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=PetaleGATR,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GAT{nb_neighbor}_{eval_id}",
                                      hps={},
                                      n_trials=0,
                                      evaluation_metrics=evaluation_metrics,
                                      fixed_params=fixed_params,
                                      fixed_params_update_function=update_fixed_params,
                                      feature_selector=feature_selector,
                                      save_hps_importance=True,
                                      save_optimization_history=True,
                                      seed=args.seed,
                                      pred_path=args.path)

                # Evaluation
                evaluator.evaluate()

        print(f"Time taken for GAT (min): {(time.time() - start)/60:.2f}")

    """
    GCN experiment
    """
    if args.gcn:

        # Start timer
        start = time.time()

        # Update of the hyperparameters
        ms_hps.GCNHPS[GCNHP.RHO.name] = args.rho

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'max_epochs': args.epochs,
                    'patience': args.patience,
                    **ms_hps.GCNHPS}

        for nb_neighbor in args.degree:

            # We change the type from str to int
            nb_neigh = int(nb_neighbor)

            # We set the conditional column
            cond_cat_col = SEX if args.conditional_column else None

            # We set the distance computations options
            GCN_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

            for prefix, w_sim in GCN_options:

                # Creation of the dataset
                dataset = PetaleKGNNDataset(df, target, k=nb_neigh,
                                            weighted_similarity=w_sim,
                                            cont_cols=cont_cols, cat_cols=cat_cols,
                                            conditional_cat_col=cond_cat_col, classification=False)

                # Saving of the fixed params of GCN
                fixed_params = update_fixed_params(dataset)

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=PetaleGCNR,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GCN{nb_neighbor}_{eval_id}",
                                      hps={},
                                      n_trials=0,
                                      evaluation_metrics=evaluation_metrics,
                                      fixed_params=fixed_params,
                                      fixed_params_update_function=update_fixed_params,
                                      feature_selector=feature_selector,
                                      save_hps_importance=True,
                                      save_optimization_history=True,
                                      seed=args.seed,
                                      pred_path=args.path)

                # Evaluation
                evaluator.evaluate()

        print(f"Time taken for GCN (min): {(time.time() - start)/60:.2f}")
