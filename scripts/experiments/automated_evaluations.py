"""
Filename: automated_evaluations.py

Authors: Nicolas Raymond

Description: This file is used to execute all the model comparisons
             made on the VO2 peak dataset

Date of last modification : 2022/07/27
"""
import sys
import time

from os.path import dirname, realpath
from copy import deepcopy

NB_TRIALS = 200

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append((dirname(dirname(dirname(realpath(__file__))))))
    from hps import search_spaces as ss
    from settings.paths import Paths
    from scripts.experiments.manual_evaluations import retrieve_arguments
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.sampling import extract_masks, get_VO2_data, push_valid_to_train
    from src.evaluation.evaluation import Evaluator
    from src.models.gat import PetaleGATR, GATHP
    from src.models.gcn import PetaleGCNR, GCNHP
    from src.models.mlp import PetaleMLPR, MLPHP
    from src.models.random_forest import PetaleRFR
    from src.models.xgboost_ import PetaleXGBR
    from src.utils.hyperparameters import Range
    from src.utils.metrics import AbsoluteError, Pearson, RootMeanSquaredError, SpearmanR, SquaredError

    # Arguments parsing
    args = retrieve_arguments()

    # Initialization of a data manager
    manager = PetaleDataManager() if not args.from_csv else None

    # We extract needed data
    df, target, cont_cols, cat_cols = get_VO2_data(manager)

    # We filter baselines variables if needed
    if args.remove_walk_variables:
        df.drop([TDM6_HR_END, TDM6_DIST], axis=1, inplace=True)
        cont_cols = [c for c in cont_cols if c not in [TDM6_HR_END, TDM6_DIST]]

    # Extraction of masks
    masks = extract_masks(Paths.VO2_MASK, k=args.nb_outer_splits, l=args.nb_inner_splits)
    push_valid_to_train(masks)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [AbsoluteError(), SpearmanR(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Initialization of a feature selector
    if args.feature_selection:
        feature_selector = FeatureSelector(threshold=[0.01], cumulative_imp=[False], seed=args.seed)
    else:
        feature_selector = None

    # We save the string that will help identify evaluations
    eval_id = "vo2_automated"
    if args.remove_walk_variables:
        eval_id += "_nw"
    if args.rho > 0:
        eval_id += "_sam"
        sam_search_space = {Range.MIN: 0, Range.MAX: args.rho}  # Sharpness-Aware Minimization search space
    else:
        sam_search_space = {Range.VALUE: 0}

    # We start a timer for the whole experiment
    first_start = time.time()

    """
    Random Forest experiment
    """
    if args.random_forest:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleRFR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"RF_{eval_id}",
                              hps=ss.RF_HPS,
                              n_trials=NB_TRIALS,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for Random Forest (minutes): {(time.time() - start) / 60:.2f}")

    """
    XGBoost experiment
    """
    if args.xg_boost:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleXGBR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"XGBoost_{eval_id}",
                              hps=ss.XGBOOST_HPS,
                              n_trials=NB_TRIALS,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for XGBoost (minutes): {(time.time() - start) / 60:.2f}")

    """
    MLP experiment
    """
    if args.mlp:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of the fixed params of MLP
        fixed_params = update_fixed_params(dataset)

        # Update of the hyperparameters
        ss.MLP_HPS[MLPHP.RHO.name] = sam_search_space
        cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
        ss.MLP_HPS[MLPHP.N_UNIT.name] = {Range.VALUE: int((len(cont_cols) + cat_sizes_sum)/2)}

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"MLP_{eval_id}",
                              hps=ss.MLP_HPS,
                              n_trials=NB_TRIALS,
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

        print(f"Time taken for MLP (minutes): {(time.time() - start) / 60:.2f}")

    """
    ENET experiment
    """
    if args.enet:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of the fixed params of ENET
        fixed_params = update_fixed_params(dataset)

        # Update of the hyperparameters
        ss.ENET_HPS[MLPHP.RHO.name] = sam_search_space

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"enet_{eval_id}",
                              hps=ss.ENET_HPS,
                              n_trials=NB_TRIALS,
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

        print(f"Time taken for enet (minutes): {(time.time() - start) / 60:.2f}")

    """
    GAT experiment
    """
    if args.gat:

        # Start timer
        start = time.time()

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
                                            conditional_cat_col=cond_cat_col, classification=False)

                # Update of hyperparameter
                cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
                ss.GATHPS[GATHP.HIDDEN_SIZE.name] = {Range.VALUE: int((len(cont_cols) + cat_sizes_sum)/2)}

                # Creation of a function to update fixed params
                def update_fixed_params(dts):
                    return {'num_cont_col': len(dts.cont_idx),
                            'cat_idx': dts.cat_idx,
                            'cat_sizes': dts.cat_sizes,
                            'cat_emb_sizes': dts.cat_sizes,
                            'max_epochs': args.epochs,
                            'patience': args.patience}

                # Saving of the fixed params pf GAT
                fixed_params = update_fixed_params(dataset)

                # Update of the hyperparameters
                ss.GATHPS[GATHP.RHO.name] = sam_search_space

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=PetaleGATR,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GAT{nb_neighbor}_{eval_id}",
                                      hps=ss.GATHPS,
                                      n_trials=NB_TRIALS,
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

        print(f"Time taken for GAT (minutes): {(time.time() - start) / 60:.2f}")

    """
    GCN experiment
    """
    if args.gcn:

        # Start timer
        start = time.time()

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

                # Update of hyperparameter
                cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
                ss.GCNHPS[GCNHP.HIDDEN_SIZE.name] = {Range.VALUE: int((len(cont_cols) + cat_sizes_sum)/2)}

                # Creation of a function to update fixed params
                def update_fixed_params(dts):
                    return {'num_cont_col': len(dts.cont_idx),
                            'cat_idx': dts.cat_idx,
                            'cat_sizes': dts.cat_sizes,
                            'cat_emb_sizes': dts.cat_sizes,
                            'max_epochs': args.epochs,
                            'patience': args.patience}

                # Saving of the fixed params of GCN
                fixed_params = update_fixed_params(dataset)

                # Update of the hyperparameters
                ss.GCNHPS[GCNHP.RHO.name] = sam_search_space

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=PetaleGCNR,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GCN{nb_neighbor}_{eval_id}",
                                      hps=ss.GCNHPS,
                                      n_trials=NB_TRIALS,
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

        print(f"Time taken for GCN (minutes): {(time.time() - start) / 60:.2f}")

    print(f"Overall time (minutes): {(time.time() - first_start) / 60:.2f}")
