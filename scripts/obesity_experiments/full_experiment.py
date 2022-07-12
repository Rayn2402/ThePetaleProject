"""
Filename: full_experiment.py

Author: Nicolas Raymond

Description: Script used to run obesity experiments automated
             hyperparameter optimization.

Date of last modification: 2022/07/11
"""

import sys
from copy import deepcopy
from os.path import dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from hps.search_spaces import ENET_HPS, ENET_GGE_HPS, GATHPS, GCNHPS, GGEHPS, MLP_HPS, RF_HPS, XGBOOST_HPS
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.sampling import get_obesity_data, extract_masks, push_valid_to_train
    from src.models.blocks.genes_signature_block import GeneEncoder, GeneGraphEncoder, GeneGraphAttentionEncoder
    from src.models.gat import PetaleBinaryGATC, PetaleGATR, GATHP
    from src.models.gge import PetaleGGE
    from src.models.gcn import PetaleBinaryGCNC, PetaleGCNR, GCNHP
    from src.models.mlp import PetaleBinaryMLPC, PetaleMLPR, MLPHP
    from src.models.random_forest import PetaleBinaryRFC, PetaleRFR, RandomForestHP
    from src.models.xgboost_ import PetaleBinaryXGBC, PetaleXGBR, XGBoostHP
    from src.training.evaluation import Evaluator
    from src.utils.argparsers import obesity_experiment_parser
    from src.utils.hyperparameters import Range
    from src.utils.score_metrics import AbsoluteError, ConcordanceIndex, Pearson, \
        RootMeanSquaredError, SquaredError, AUC, BinaryCrossEntropy, BinaryBalancedAccuracy, \
        Sensitivity, Specificity, BalancedAccuracyEntropyRatio, Reduction
    from time import time
    from typing import Dict, List, Optional

    # Arguments parsing
    args = obesity_experiment_parser()

    # Initialization of DataManager
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
        evaluation_metrics = [AUC(), BinaryBalancedAccuracy(), Sensitivity(),
                              Specificity(), BinaryCrossEntropy(),
                              BalancedAccuracyEntropyRatio(reduction=Reduction.GEO_MEAN)]
    else:
        evaluation_metrics = [AbsoluteError(), ConcordanceIndex(), Pearson(), SquaredError(),
                              RootMeanSquaredError()]

    # Initialization of feature selector
    if args.feature_selection:
        if args.baselines and args.genomics:
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
    experiment_id = f"obesity_automated"
    if args.classification:
        experiment_id += "_c"
    if args.genomic:
        experiment_id += "_snps"
    if args.rho > 0:
        experiment_id += "_sam"
        sam_search_space = {Range.MIN: 0.05, Range.MAX: 2}
    else:
        sam_search_space = {Range.VALUE: 0}

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
            RF_HPS[RandomForestHP.WEIGHT.name] = {Range.VALUE: 0.5}
        else:
            constructor = PetaleRFR

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"RandomForest_{experiment_id}",
                              hps=free_hps.RF_HPS,
                              n_trials=200,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for Random Forest (minutes): ", round((time() - start) / 60, 2))

    """
    XGBoost experiment
    """
    if args.xg_boost:

        # Start timer
        start = time()

        # Creation of dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, classification=args.classification,
                                feature_selection_groups=[gene_cols])

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryXGBC
            XGBOOST_HPS[XGBoostHP.WEIGHT.name] = {Range.VALUE: 0.5}
        else:
            constructor = PetaleXGBR

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"XGBoost_{experiment_id}",
                              hps=free_hps.XGBOOST_HPS,
                              n_trials=200,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed,
                              pred_path=args.path)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for XGBoost (minutes): ", round((time() - start) / 60, 2))

    """
    MLP experiment
    """
    if args.mlp:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols, to_tensor=True,
                                classification=args.classification, feature_selection_groups=[gene_cols])

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryMLPC
            MLP_HPS[MLPHP.WEIGHT.name] = {Range.VALUE: 0.5}
        else:
            constructor = PetaleMLPR

        # Update of hyperparameters
        if args.enable_sam:
            MLP_HPS[MLPHP.RHO.name] = sam_search_space

        cat_sizes_sum = sum(dataset.cat_sizes) if dataset.cat_sizes is not None else 0
        MLP_HPS[MLPHP.N_UNIT.name] = int((len(cont_cols) + cat_sizes_sum) / 2)

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of fixed_params for MLP
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"MLP_{experiment_id}",
                              hps=free_hps.MLP_HPS,
                              n_trials=200,
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

        print("Time Taken for MLP (minutes): ", round((time() - start) / 60, 2))

    """
    ENET experiment
    """
    if args.enet:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                to_tensor=True, classification=args.classification,
                                feature_selection_groups=[gene_cols])

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryMLPC
            MLP_HPS[MLPHP.WEIGHT.name] = {Range.VALUE: 0.5}
        else:
            constructor = PetaleMLPR

        # Update of hyperparameters
        if args.enable_sam:
            ENET_HPS[MLPHP.RHO.name] = sam_search_space

        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of fixed_params for ENET
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"enet_{experiment_id}",
                              hps=free_hps.ENET_HPS,
                              n_trials=200,
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

        print("Time Taken for ENET (minutes): ", round((time() - start) / 60, 2))

    """
    GGE experiment
    """
    if args.gge and genes:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=gene_cols, to_tensor=True,
                                classification=args.classification, feature_selection_groups=[gene_cols])

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

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryMLPC
            MLP_HPS[MLPHP.WEIGHT.name] = {Range.VALUE: 0.5}
        else:
            constructor = PetaleMLPR

        # Update of hyperparameters
        if args.enable_sam:
            ENET_GGE_HPS[MLPHP.RHO.name] = sam_search_space

        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor}

        # Saving of fixed_params for GGE + ENET
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggeEnet_{experiment_id}",
                              hps=free_hps.ENET_GGE_HPS,
                              n_trials=200,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for GGE (minutes): ", round((time() - start) / 60, 2))

    """
    GGAE experiment
    """
    if args.ggae and genes:

        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=gene_cols, to_tensor=True,
                                classification=args.classification, feature_selection_groups=[gene_cols])

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

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryMLPC
            MLP_HPS[MLPHP.WEIGHT.name] = {Range.VALUE: 0.5}
        else:
            constructor = PetaleMLPR

        # Update of hyperparameters
        if args.enable_sam:
            ENET_GGE_HPS[MLPHP.RHO.name] = sam_search_space

        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor}

        # Saving of fixed_params for GGAE + ENET
        fixed_params = update_fixed_params(dataset)

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=constructor,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggaeEnet_{experiment_id}",
                              hps=free_hps.ENET_GGE_HPS,
                              n_trials=200,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for GGAE (minutes): ", round((time() - start) / 60, 2))

    """
    GAT experiment
    """
    if args.gat:

        # Start timer
        start = time()

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryGATC
            GATHPS[GATHP.WEIGHT.name] = {Range.VALUE: 0.5}
        else:
            constructor = PetaleGATR

        # Update of hyperparameters
        if args.enable_sam:
            GATHPS[GATHP.RHO.name] = sam_search_space

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'max_epochs': 500,
                    'patience': 50}

        for nb_neighbor in args.degree:

            # We change the type from str to int
            nb_neigh = int(nb_neighbor)

            # We set the conditional column
            cond_cat_col = SEX if args.conditional_column else None

            # We set the distance computations options
            GAT_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

            for prefix, w_sim in GAT_options:
                dataset = PetaleKGNNDataset(df, target, k=nb_neigh,
                                            weighted_similarity=w_sim,
                                            cont_cols=cont_cols, cat_cols=cat_cols,
                                            conditional_cat_col=cond_cat_col,
                                            classification=args.classification,
                                            feature_selection_groups=[gene_cols])

                # Saving of original fixed params for GAT
                fixed_params = update_fixed_params(dataset)

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=constructor,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GAT{nb_neighbor}_{experiment_id}",
                                      hps=free_hps.GATHPS,
                                      n_trials=200,
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

        print("Time Taken for GAT (minutes): ", round((time() - start) / 60, 2))

    """
    GCN experiment
    """
    if args.gcn:

        # Start timer
        start = time()

        # Constructor selection
        if args.classification:
            constructor = PetaleBinaryGCNC
            GCNHPS[GCNHP.WEIGHT.name] = {Range.VALUE: 0.5}
        else:
            constructor = PetaleGCNR

        # Update of hyperparameters
        if args.enable_sam:
            GCNHPS[GCNHP.RHO.name] = sam_search_space

        # Creation of function to update fixed params
        def update_fixed_params(dts):
            return {'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'max_epochs': 500,
                    'patience': 50}

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
                                            classification=args.classification,
                                            feature_selection_groups=[gene_cols])

                # Saving of original fixed params for GCN
                fixed_params = update_fixed_params(dataset)

                # Creation of the evaluator
                evaluator = Evaluator(model_constructor=constructor,
                                      dataset=dataset,
                                      masks=masks,
                                      evaluation_name=f"{prefix}GCN{nb_neighbor}_{experiment_id}",
                                      hps=free_hps.GCNHPS,
                                      n_trials=200,
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

        print("Time Taken for GCN (minutes): ", round((time() - start) / 60, 2))

    """
    Self supervised learning experiment with GGE
    """
    if args.ssl_gge and genes:
        # Start timer
        start = time()

        # Creation of the dataset
        dataset = PetaleDataset(df, target, cont_cols, cat_cols,
                                gene_cols=gene_cols, to_tensor=True, classification=False,
                                feature_selection_groups=[gene_cols])

        # Creation of a function to update fixed params
        def update_fixed_params(dts):
            return {'max_epochs': 500,
                    'patience': 50,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'hidden_size': 2,
                    'signature_size': args.signature_size,
                    'genes_emb_sharing': args.embedding_sharing,
                    'aggregation_method': 'avg'}

        # Saving of original fixed params for GGAE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleGGE,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"gge_{experiment_id}",
                              hps=free_hps.GGEHPS,
                              n_trials=200,
                              evaluation_metrics=[],
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              feature_selector=feature_selector,
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=args.seed)

        # Evaluation
        evaluator.evaluate()

        print("Time Taken for Self Supervised GGE (minutes): ", round((time() - start) / 60, 2))

    print("Overall time (minutes): ", round((time() - first_start) / 60, 2))
