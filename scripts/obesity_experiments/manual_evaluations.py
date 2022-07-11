"""
Filename: manual_evaluations.py

Author: Nicolas Raymond

Description: Script used to run obesity experiments using
             manually selected hyperparameters.

Date of last modification: 2022/07/11
"""
import sys
from copy import deepcopy
from os.path import dirname, realpath
from time import time
from typing import Dict, List, Optional

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from hps import manually_selected_hps as ms_hps
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.sampling import extract_masks, get_obesity_data, push_valid_to_train
    from src.models.blocks.genes_signature_block import GeneEncoder, GeneGraphEncoder, GeneGraphAttentionEncoder
    from src.models.gat import PetaleBinaryGATC, PetaleGATR, GATHP
    from src.models.gge import PetaleGGE
    from src.models.gcn import PetaleBinaryGCNC, PetaleGCNR, GCNHP
    from src.models.mlp import PetaleBinaryMLPC, PetaleMLPR, MLPHP
    from src.models.random_forest import PetaleBinaryRFC, PetaleRFR, RandomForestHP
    from src.models.xgboost_ import PetaleBinaryXGBC, PetaleXGBR, XGBoostHP
    from src.training.evaluation import Evaluator
    from src.utils.argparsers import obesity_experiment_parser
    from src.utils import score_metrics as sm

    # Arguments parsing
    args = obesity_experiment_parser()

    # Initialization of a data manager
    manager = PetaleDataManager()

    # We extract needed data
    df, target, cont_cols, cat_cols = get_obesity_data(data_manager=manager,
                                                       genomics=args.genomics,
                                                       baselines=args.baselines,
                                                       classification=args.classification,
                                                       holdout=args.holdout)
    # Extraction of masks
    if args.holdout:
        masks = extract_masks(Paths.OBESITY_HOLDOUT_MASK, k=1, l=10)
    else:
        masks = extract_masks(Paths.OBESITY_MASK, k=args.nb_outer_splits, l=args.nb_inner_splits)

    masks_without_val = deepcopy(masks)
    push_valid_to_train(masks_without_val)

    # Initialization of the dictionary containing the evaluation metrics
    if args.classification:
        evaluation_metrics = [sm.AUC(), sm.BinaryBalancedAccuracy(), sm.Sensitivity(),
                              sm.Specificity(), sm.BinaryCrossEntropy(),
                              sm.BalancedAccuracyEntropyRatio(reduction=sm.Reduction.GEO_MEAN)]
    else:
        evaluation_metrics = [sm.AbsoluteError(), sm.ConcordanceIndex(), sm.Pearson(),
                              sm.SquaredError(), sm.RootMeanSquaredError()]

    # Initialization of feature selector
    if args.feature_selection:
        if args.baselines and args.genomic:
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
    experiment_id = f"obesity_manual"
    if args.classification:
        experiment_id += "_c"
    if args.genomic:
        experiment_id += "_snps"
    if args.rho > 0:
        experiment_id += "_sam"

    # We start a timer for the whole experiment
    first_start = time()

    """
    Random Forest experiment
    """
    if args.random_forest:

        # Start timer
        start = time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=args.classification,
                                feature_selection_groups=[OBESITY_SNPS])

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryRFC
            ms_hps.RF_HPS[RandomForestHP.WEIGHT.name] = 0.5
        else:
            constructor = PetaleRFR

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"RF_{experiment_id}",
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

        print(f"Time taken for Random Forest (minutes): {(time() - start) / 60:.2f}")

    """
    XGBoost experiment
    """
    if args.xg_boost:

        # Start timer
        start = time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=args.classification,
                                feature_selection_groups=[OBESITY_SNPS])

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryXGBC
            ms_hps.XGBOOST_HPS[XGBoostHP.WEIGHT.name] = 0.5
        else:
            constructor = PetaleXGBR

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"XGBoost_{experiment_id}",
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

        print(f"Time taken for XGBoost (minutes): {(time() - start) / 60:.2f}")

    """
    MLP experiment
    """
    if args.mlp:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True,
                                classification=args.classification, feature_selection_groups=[OBESITY_SNPS])

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryMLPC
            ms_hps.MLP_HPS[MLPHP.WEIGHT.name] = 0.5
        else:
            constructor = PetaleMLPR

        # Update of the hyperparameters
        ms_hps.MLP_HPS[MLPHP.RHO.name] = args.rho
        cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
        ms_hps.MLP_HPS[MLPHP.N_UNIT.name] = int((len(cont_cols) + cat_sizes_sum) / 2)

        # Creation of a function to update the fixed params
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

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"MLP_{experiment_id}",
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

        print(f"Time taken for MLP (minutes): {(time() - start) / 60:.2f}")

    """
    ENET experiment
    """
    if args.enet:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                to_tensor=True, classification=args.classification,
                                feature_selection_groups=[OBESITY_SNPS])

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryMLPC
            ms_hps.MLP_HPS[MLPHP.WEIGHT.name] = 0.5
        else:
            constructor = PetaleMLPR

        # Update of the hyperparameters
        ms_hps.ENET_HPS[MLPHP.RHO.name] = args.rho

        # Creation of a function to update the fixed params
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
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"enet_{experiment_id}",
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

        print(f"Time taken for enet (minutes): {(time() - start) / 60:.2f}")

    """
    GGE experiment
    """
    if args.gge and args.genomics:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=OBESITY_SNPS, to_tensor=True,
                                classification=args.classification, feature_selection_groups=[OBESITY_SNPS])

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


        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryMLPC
            ms_hps.ENET_GGE_HPS[MLPHP.WEIGHT.name] = 0.5
        else:
            constructor = PetaleMLPR

        # Update of the hyperparameters
        ms_hps.ENET_GGE_HPS[MLPHP.RHO.name] = args.rho

        # Creation of a function to update the fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor,
                    **ms_hps.ENET_GGE_HPS}


        # Saving of fixed_params for ENET + GGE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggeEnet_{experiment_id}",
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

        print(f"Time taken for enet (minutes): {(time() - start) / 60:.2f}")

    """
    GGAE experiment
    """
    if args.ggae and args.genomics:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=OBESITY_SNPS, to_tensor=True,
                                classification=args.classification, feature_selection_groups=[OBESITY_SNPS])

        # Creation of a function to update the fixed params
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

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryMLPC
            ms_hps.ENET_GGE_HPS[MLPHP.WEIGHT.name] = 0.5
        else:
            constructor = PetaleMLPR

        # Update of the hyperparameters
        ms_hps.ENET_GGE_HPS[MLPHP.RHO.name] = args.rho

        # Creation of a function to update the fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor,
                    **ms_hps.ENET_GGE_HPS}


        # Saving of the fixed_params for ENET + GGAE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggaeEnet_{experiment_id}",
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

        print(f"Time taken for GGAE (minutes): {(time() - start) / 60:.2f}")

    """
    GAT experiment
    """
    if args.gat:

        # Start timer
        start = time()

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryGATC
            ms_hps.GATHPS[GATHP.WEIGHT.name] = 0.5
        else:
            constructor = PetaleGATR

        # Update of the hyperparameters
        ms_hps.GATHPS[GATHP.RHO.name] = args.rho

        # Creation of a function to update the fixed params
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
                                            classification=args.classification, feature_selection_groups=[OBESITY_SNPS])

                # Saving of the fixed params of GAT
                fixed_params = update_fixed_params(dataset)

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=constructor,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GAT{nb_neighbor}_{experiment_id}",
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

        print(f"Time taken for GAT (minutes): {(time() - start) / 60:.2f}")

    """
    GCN experiment
    """
    if args.gcn:

        # Start timer
        start = time()

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryGCNC
            ms_hps.GCNHPS[GCNHP.WEIGHT.name] = 0.5
        else:
            constructor = PetaleGCNR

        # Update of the hyperparameters
        ms_hps.GCNHPS[GCNHP.RHO.name] = args.rho

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
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
                                            conditional_cat_col=cond_cat_col,
                                            classification=args.classification, feature_selection_groups=[OBESITY_SNPS])

                # Saving of the fixed params of GCN
                fixed_params = update_fixed_params(dataset)

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=constructor,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GCN{nb_neighbor}_{experiment_id}",
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

        print(f"Time taken for GCN (minutes): {(time() - start) / 60:.2f}")

    """
    Self supervised learning experiment with GGE
    """
    if args.ssl_gge and args.genomics:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=OBESITY_SNPS, to_tensor=True, classification=False,
                                feature_selection_groups=[OBESITY_SNPS])

        # Creation of a function to update the fixed params
        def update_fixed_params(dts):
            return {'max_epochs': args.epochs,
                    'patience': args.patience,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'hidden_size': 3,
                    'signature_size': args.signature_size,
                    'genes_emb_sharing': args.embedding_sharing,
                    'aggregation_method': 'avg',
                    **ms_hps.GGEHPS}


        # Saving of the fixed params of GGE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleGGE,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"gge_{experiment_id}",
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

        print(f"Time taken for SSL GGE (minutes): {(time() - start) / 60:.2f}")

    print(f"Overall time (minutes): {(time() - first_start) / 60:.2f}")

