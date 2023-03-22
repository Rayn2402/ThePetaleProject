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
from typing import Dict, List, Optional

NB_TRIALS = 200

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
    from hps import search_spaces as ss
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.sampling import extract_masks, get_VO2_data, push_valid_to_train
    from src.models.blocks.genes_signature_block import GeneEncoder, GeneGraphEncoder, GeneGraphAttentionEncoder
    from src.models.gat import PetaleGATR, GATHP
    from src.models.gcn import PetaleGCNR, GCNHP
    from src.models.gge import PetaleGGE
    from src.models.mlp import PetaleMLPR, MLPHP
    from src.models.random_forest import PetaleRFR
    from src.models.xgboost_ import PetaleXGBR
    from src.evaluation.evaluation import Evaluator
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.argparsers import VO2_experiment_parser
    from src.utils.hyperparameters import Range
    from src.utils.metrics import AbsoluteError, ConcordanceIndex, Pearson, RootMeanSquaredError, SquaredError

    # Arguments parsing
    args = VO2_experiment_parser()

    # Initialization of a data manager
    manager = PetaleDataManager() if not args.from_csv else None

    # We extract needed data
    df, target, cont_cols, cat_cols = get_VO2_data(manager,
                                                   baselines=args.baselines,
                                                   genomics=args.genomics,
                                                   sex=args.sex,
                                                   holdout=args.holdout)
    # We modify SNPs list according to the given arguments
    VO2_SNPS = None if not args.genomics else VO2_SNPS

    # We filter baselines variables if needed
    if args.baselines and args.remove_walk_variables:
        df.drop([TDM6_HR_END, TDM6_DIST], axis=1, inplace=True)
        cont_cols = [c for c in cont_cols if c not in [TDM6_HR_END, TDM6_DIST]]

    # Extraction of masks
    if args.holdout:
        masks = extract_masks(Paths.VO2_HOLDOUT_MASK, k=1, l=10)
    else:
        masks = extract_masks(Paths.VO2_MASK, k=args.nb_outer_splits, l=args.nb_inner_splits)

    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [AbsoluteError(), ConcordanceIndex(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Initialization of a feature selector
    if args.feature_selection:
        if args.genomics and args.baselines:
            feature_selector = FeatureSelector(threshold=[0.01, 0.01],
                                               cumulative_imp=[False, False],
                                               seed=args.seed)
        else:
            feature_selector = FeatureSelector(threshold=[0.01],
                                               cumulative_imp=[False],
                                               seed=args.seed)
    else:
        feature_selector = None

    # We save the string that will help identify evaluations
    eval_id = "vo2_automated"
    if args.baselines:
        eval_id += "_b"
        if args.remove_walk_variables:
            eval_id += "_nw"
    if args.genomics:
        eval_id += "_snps"
    if args.sex:
        eval_id += "_sex"
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
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                classification=False, feature_selection_groups=[VO2_SNPS])

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleRFR,
                              dataset=dataset,
                              masks=masks_without_val,
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
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                classification=False, feature_selection_groups=[VO2_SNPS])

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleXGBR,
                              dataset=dataset,
                              masks=masks_without_val,
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
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True,
                                classification=False, feature_selection_groups=[VO2_SNPS])

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
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                to_tensor=True, classification=False,
                                feature_selection_groups=[VO2_SNPS])

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
    GGE experiment
    """
    if args.gge and args.genomics:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=VO2_SNPS, to_tensor=True,
                                classification=False, feature_selection_groups=[VO2_SNPS])

        # Creation of a function that builds a Gene Graph Encoder
        def gene_encoder_constructor(gene_idx_groups: Optional[Dict[str, List[int]]],
                                     dropout: float) -> GeneEncoder:
            """
            Builds a Gene Graph Encoder

            Args:
                gene_idx_groups: dictionary where keys are names of chromosomes pairs and values
                                 are list of idx referring to columns of SNPs associated to
                                 the chromosomes pairs
                dropout: dropout probability

            Returns: GeneEncoder
            """

            return GeneGraphEncoder(gene_idx_groups=gene_idx_groups,
                                    genes_emb_sharing=args.embedding_sharing,
                                    dropout=dropout,
                                    signature_size=args.signature_size)


        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor}


        # Saving of the fixed params of GGE + ENET
        fixed_params = update_fixed_params(dataset)

        # Update of the hyperparameters
        ss.ENET_GGE_HPS[MLPHP.RHO.name] = sam_search_space

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggeEnet_{eval_id}",
                              hps=ss.ENET_GGE_HPS,
                              n_trials=NB_TRIALS,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for GGE (minutes): {(time.time() - start) / 60:.2f}")

    """
    GGAE experiment
    """
    if args.ggae and args.genomics:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=VO2_SNPS, to_tensor=True,
                                classification=False, feature_selection_groups=[VO2_SNPS])

        # Creation of a function that builds a Gene Graph Attention Encoder
        def gene_encoder_constructor(gene_idx_groups: Optional[Dict[str, List[int]]],
                                     dropout: float) -> GeneEncoder:
            """
            Builds a Gene Graph Attention Encoder

            Args:
                gene_idx_groups: dictionary where keys are names of chromosomes pairs and values
                                 are list of idx referring to columns of SNPs associated to
                                 the chromosomes pairs
                dropout: dropout probability

            Returns: GeneEncoder
            """

            return GeneGraphAttentionEncoder(gene_idx_groups=gene_idx_groups,
                                             genes_emb_sharing=args.embedding_sharing,
                                             dropout=dropout,
                                             signature_size=args.signature_size)


        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor}


        # Saving of fixed_params for GGAE + ENET
        fixed_params = update_fixed_params(dataset)

        # Update of the hyperparameters
        ss.ENET_GGE_HPS[MLPHP.RHO.name] = sam_search_space

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggaeEnet_{eval_id}",
                              hps=ss.ENET_GGE_HPS,
                              n_trials=NB_TRIALS,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for GGAE (minutes): {(time.time() - start) / 60:.2f}")

    """
    GAT experiment
    """
    if args.gat and args.baselines:

        # Start timer
        start = time.time()

        for nb_neighbor in args.degree:

            # We change the type from str to int
            nb_neigh = int(nb_neighbor)

            # We set the conditional column
            cond_cat_col = SEX if args.sex and args.conditional_column else None

            # We set the distance computations options
            GAT_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

            for prefix, w_sim in GAT_options:

                # Creation of the dataset
                dataset = PetaleKGNNDataset(df, target, k=nb_neigh,
                                            weighted_similarity=w_sim,
                                            cont_cols=cont_cols, cat_cols=cat_cols,
                                            conditional_cat_col=cond_cat_col, classification=False,
                                            feature_selection_groups=[VO2_SNPS])

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
    if args.gcn and args.baselines:

        # Start timer
        start = time.time()

        for nb_neighbor in args.degree:

            # We change the type from str to int
            nb_neigh = int(nb_neighbor)

            # We set the conditional column
            cond_cat_col = SEX if args.sex and args.conditional_column else None

            # We set the distance computations options
            GCN_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

            for prefix, w_sim in GCN_options:

                # Creation of the dataset
                dataset = PetaleKGNNDataset(df, target, k=nb_neigh,
                                            weighted_similarity=w_sim,
                                            cont_cols=cont_cols, cat_cols=cat_cols,
                                            conditional_cat_col=cond_cat_col, classification=False,
                                            feature_selection_groups=[VO2_SNPS])

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

    """
    Self supervised learning experiment with GGE
    """
    if args.ssl_gge and args.genomics:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=VO2_SNPS, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epoch,
                    'patience': args.patience,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'hidden_size': 3,
                    'signature_size': args.signature_size,
                    'genes_emb_sharing': args.embedding_sharing,
                    'aggregation_method': 'avg'}

        # Saving of the fixed params of GGE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleGGE,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"gge_{eval_id}",
                              hps=ss.GGEHPS,
                              n_trials=NB_TRIALS,
                              evaluation_metrics=[],
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print(f"Time taken for SSL GGE (minutes): {(time.time() - start) / 60:.2f}")

    print(f"Overall time (minutes): {(time.time() - first_start) / 60:.2f}")
