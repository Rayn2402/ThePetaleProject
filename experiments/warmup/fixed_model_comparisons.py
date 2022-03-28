"""
Filename: fixed_model_comparisons.py

Author: Nicolas Raymond

Description: This file is a script used to run warmup experiments using fixed
             hyperparameters.

Date of last modification: 2022/03/28
"""

import sys
import time

from os.path import dirname, realpath
from copy import deepcopy
from typing import Dict, List, Optional


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from hps.fixed_hps import ENET_HPS, ENET_GGE_HPS, GATHPS, GGEHPS, MLP_HPS, RF_HPS,XGBOOST_HPS
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.sampling import extract_masks, GeneChoice, get_warmup_data, push_valid_to_train
    from src.models.blocks.genes_signature_block import GeneEncoder, GeneGraphEncoder, GeneGraphAttentionEncoder
    from src.models.gat import PetaleGATR, GATHP
    from src.models.gge import PetaleGGE
    from src.models.mlp import PetaleMLPR, MLPHP
    from src.models.random_forest import PetaleRFR
    from src.models.xgboost_ import PetaleXGBR, XGBoostHP
    from src.training.evaluation import Evaluator
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.argparsers import warmup_experiment_parser
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError, SquaredError

    # Arguments parsing
    args = warmup_experiment_parser()

    # Initialization of DataManager and sampler
    manager = PetaleDataManager()

    # We extract needed data
    if args.genes_subgroup:
        genes_selection = GeneChoice.SIGNIFICANT
        gene_cols = SIGNIFICANT_CHROM_POS_WARMUP
    elif args.all_genes:
        genes_selection = GeneChoice.ALL
        gene_cols = ALL_CHROM_POS_WARMUP
    else:
        genes_selection = None
        gene_cols = None

    genes = True if genes_selection is not None else False
    df, target, cont_cols, cat_cols = get_warmup_data(manager,
                                                      baselines=args.baselines,
                                                      genes=genes_selection,
                                                      sex=args.sex)
    # We filter gene variables if needed
    if args.single_gen:
        removed_genes = [g for g in gene_cols if g != '7_45932669']
        df.drop(removed_genes, axis=1, inplace=True)
        cat_cols = [c for c in cat_cols if c not in removed_genes]

    # We filter baselines variables if needed
    if args.baselines and args.remove_walk_variables:
        df.drop([TDM6_HR_END, TDM6_DIST], axis=1, inplace=True)
        cont_cols = [c for c in cont_cols if c not in [TDM6_HR_END, TDM6_DIST]]

    # Extraction of masks
    masks = extract_masks(Paths.WARMUP_MASK, k=args.nb_outer_splits, l=args.nb_inner_splits)
    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [AbsoluteError(), Pearson(), SquaredError(), RootMeanSquaredError()]

    # Initialization of feature selector
    if args.feature_selection:
        feature_selector = FeatureSelector(importance_threshold=0.90, seed=args.seed)
    else:
        feature_selector = None

    # We save the string that will help identify evaluations
    eval_id = "_fixedhps"
    if args.baselines:
        eval_id += "_baselines"
        if args.remove_walk_variables:
            eval_id += "_nw"
    if genes:
        if args.all_genes:
            eval_id += "_gen2"
        else:
            eval_id += "_gen1"
    if args.sex:
        eval_id += "_sex"
    if args.enable_sam:
        eval_id += "_sam"

    # We save the Sharpness-Aware Minimization search space
    sam_value = 0.05

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
                              masks=masks_without_val,
                              evaluation_name=f"RandomForest_warmup{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=RF_HPS,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for Random Forest (minutes): ", round((time.time() - start) / 60, 2))

    """
    XGBoost experiment
    """
    if args.xg_boost:

        # Start timer
        start = time.time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=False)

        # Hyperparameters update
        if genes and not args.single_gen:
            XGBOOST_HPS[XGBoostHP.ALPHA.name] = 0

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleXGBR,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"XGBoost_warmup{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=XGBOOST_HPS,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

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
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    **MLP_HPS}

        # Saving of fixed_params for MLP
        fixed_params = update_fixed_params(dataset)

        # Update of hyperparameters
        if args.enable_sam:
            MLP_HPS[MLPHP.RHO.name] = sam_value

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"MLP_warmup{eval_id}",
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

        print("Time Taken for MLP (minutes): ", round((time.time() - start) / 60, 2))

    """
    ENET experiment
    """
    if args.enet:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                to_tensor=True, classification=False)

        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    **ENET_HPS}

        # Saving of fixed_params for ENET
        fixed_params = update_fixed_params(dataset)

        # Update of hyperparameters
        if args.enable_sam:
            ENET_HPS[MLPHP.RHO.name] = sam_value
        if genes and not args.single_gen:
            ENET_HPS[MLPHP.ALPHA.name] = 0

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"enet_warmup{eval_id}",
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

        print("Time Taken for ENET (minutes): ", round((time.time() - start) / 60, 2))

    """
    GGE experiment
    """
    if args.gge and genes:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=gene_cols, to_tensor=True, classification=False)

        def gene_encoder_constructor(gene_idx_groups: Optional[Dict[str, List[int]]],
                                     dropout: float) -> GeneEncoder:
            """
            Builds a GeneGraphEncoder

            Args:
                gene_idx_groups: dictionary where keys are names of chromosomes and values
                                 are list of idx referring to columns of genes associated to
                                 the chromosome
                dropout: dropout probability

            Returns: GeneEncoder
            """

            return GeneGraphEncoder(gene_idx_groups=gene_idx_groups,
                                    genes_emb_sharing=args.embedding_sharing,
                                    dropout=dropout,
                                    signature_size=args.signature_size)

        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor,
                    **ENET_GGE_HPS}


        # Saving of fixed_params for GGE + ENET
        fixed_params = update_fixed_params(dataset)

        # Update of hyperparameters
        if args.enable_sam:
            ENET_GGE_HPS[MLPHP.RHO.name] = sam_value

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggeEnet_warmup{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for GGE (minutes): ", round((time.time() - start) / 60, 2))

    """
    GGAE experiment
    """
    if args.ggae and genes:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=gene_cols, to_tensor=True, classification=False)


        def gene_encoder_constructor(gene_idx_groups: Optional[Dict[str, List[int]]],
                                     dropout: float) -> GeneEncoder:
            """
            Builds a GeneGraphAttentionEncoder

            Args:
                gene_idx_groups: dictionary where keys are names of chromosomes and values
                                 are list of idx referring to columns of genes associated to
                                 the chromosome
                dropout: dropout probability

            Returns: GeneEncoder
            """

            return GeneGraphAttentionEncoder(gene_idx_groups=gene_idx_groups,
                                             genes_emb_sharing=args.embedding_sharing,
                                             dropout=dropout,
                                             signature_size=args.signature_size)

        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor,
                    **ENET_GGE_HPS}


        # Saving of fixed_params for GGAE + ENET
        fixed_params = update_fixed_params(dataset)

        # Update of hyperparameters
        if args.enable_sam:
            ENET_GGE_HPS[MLPHP.RHO.name] = sam_value

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggaeEnet_warmup{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for GGAE (minutes): ", round((time.time() - start) / 60, 2))

    """
    GAT experiment
    """
    if args.gat and args.baselines:

        # Start timer
        start = time.time()

        for nb_neighbor in args.degree:

            nb_neighbor = int(nb_neighbor)

            if args.sex and args.conditional_column:
                cond_cat_col = SEX
                nb_neighbor = int(nb_neighbor/2)
            else:
                cond_cat_col = None

            GAT_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

            for prefix, w_sim in GAT_options:

                # Creation of the dataset
                dataset = PetaleKGNNDataset(df, target, k=nb_neighbor,
                                            weighted_similarity=w_sim,
                                            cont_cols=cont_cols, cat_cols=cat_cols,
                                            conditional_cat_col=cond_cat_col, classification=False)

                # Creation of function to update fixed params
                def update_fixed_params(dts):
                    return {'num_cont_col': len(dts.cont_idx),
                            'cat_idx': dts.cat_idx,
                            'cat_sizes': dts.cat_sizes,
                            'cat_emb_sizes': dts.cat_sizes,
                            'max_epochs': 500,
                            'patience': 50,
                            **GATHPS}

                # Saving of original fixed params for HAN
                fixed_params = update_fixed_params(dataset)

                # Update of hyperparameters
                if args.enable_sam:
                    GATHPS[GATHP.RHO.name] = sam_value

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=PetaleGATR,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GAT{nb_neighbor}_warmup{eval_id}",
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

        print("Time Taken for GAT (minutes): ", round((time.time() - start) / 60, 2))

    """
    Self supervised learning experiment with GGAE
    """
    if args.ssl_ggae and genes:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=gene_cols, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'hidden_size': 3,
                    'signature_size': args.signature_size,
                    'genes_emb_sharing': args.embedding_sharing,
                    'aggregation_method': 'att',
                    **GGEHPS}

        # Saving of original fixed params for GGAE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleGGE,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggae_warmup{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=[],
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for Self Supervised GGAE (minutes): ", round((time.time() - start) / 60, 2))

    """
    Self supervised learning experiment with GGE
    """
    if args.ssl_gge and genes:

        # Start timer
        start = time.time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=gene_cols, to_tensor=True, classification=False)

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'hidden_size': 3,
                    'signature_size': args.signature_size,
                    'genes_emb_sharing': args.embedding_sharing,
                    'aggregation_method': 'avg',
                    **GGEHPS}

        # Saving of original fixed params for GGAE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleGGE,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"gge_warmup{eval_id}",
                              hps={},
                              n_trials=0,
                              evaluation_metrics=[],
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for Self Supervised GGE (minutes): ", round((time.time() - start) / 60, 2))

    print("Overall time (minutes): ", round((time.time() - first_start) / 60, 2))
