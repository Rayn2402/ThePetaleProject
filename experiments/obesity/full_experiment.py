"""
Filename: full_experiment.py

Authors: Nicolas Raymond

Description: This file is used to execute all the model comparisons
             made on the BMI dataset

Date of last modification : 2022/03/08
"""
import sys
import argparse
import time

from os.path import dirname, realpath
from copy import deepcopy
from typing import Dict, List, Optional


def argument_parser():
    """
    This function defines a parser that enables user to easily run different experiments
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python full_experiment.py',
                                     description="Runs all the experiments associated to the bmi dataset")

    # Nb inner split and nb outer split selection
    parser.add_argument('-k', '--nb_outer_splits', type=int, default=10,
                        help='Number of outer splits during the models evaluations')
    parser.add_argument('-l', '--nb_inner_splits', type=int, default=10,
                        help='Number of inner splits during the models evaluations')

    # Features selection
    parser.add_argument('-gen1', '--genes_subgroup', default=False, action='store_true',
                        help='True if we want to include genes if features')
    parser.add_argument('-gen2', '--all_genes', default=False, action='store_true',
                        help='True if we want to include genes if features')
    parser.add_argument('-f', '--feature_selection', default=False, action='store_true',
                        help='True if we want to apply automatic feature selection')

    # Genes encoding parameter
    parser.add_argument('-share', '--embedding_sharing', default=False, action='store_true',
                        help='True if we want to use a single entity embedding layer for all genes'
                             ' (currently only applies with genomic signature creation')

    # Models selection
    parser.add_argument('-enet', '--enet', default=False, action='store_true',
                        help='True if we want to enet experiment')
    parser.add_argument('-mlp', '--mlp', default=False, action='store_true',
                        help='True if we want to run mlp experiment')
    parser.add_argument('-rf', '--random_forest', default=False, action='store_true',
                        help='True if we want to run random forest experiment')
    parser.add_argument('-xg', '--xg_boost', default=False, action='store_true',
                        help='True if we want to run xgboost experiment')
    parser.add_argument('-gat', '--gat', default=False, action='store_true',
                        help='True if we want to run GraphAttentionNetwork experiment')
    parser.add_argument('-gge', '--gge', default=False, action='store_true',
                        help='True if we want to run GeneGraphEncoder with enet experiment')
    parser.add_argument('-ggae', '--ggae', default=False, action='store_true',
                        help='True if we want to run GeneGraphAttentionEncoder with enet experiment')

    # GAT graph construction parameters
    parser.add_argument('-w_sim', '--weighted_similarity', default=False, action='store_true',
                        help='True if we want calculate patients similarities using weighted metrics')
    parser.add_argument('-cond_col', '--conditional_column', default=False, action='store_true',
                        help='True if we want calculate to use the sex as conditional column in GAT construction')

    # Gene encoding parameter
    parser.add_argument('-sign_size', '--signature_size', type=int, default=8,
                        help='Genomic signature size')

    # Self supervised learning experiments
    parser.add_argument('-ssl_ggae', '-ssl_ggae', default=False, action='store_true',
                        help='True if we want to run self supervised learning with the GeneGraphAttentionEncoder')
    parser.add_argument('-ssl_gge', '-ssl_gge', default=False, action='store_true',
                        help='True if we want to run self supervised learning with the GeneGraphEncoder')

    # Activation of sharpness-aware minimization
    parser.add_argument('-sam', '--enable_sam', default=False, action='store_true',
                        help='True if we want to use Sharpness-Aware Minimization Optimizer')


    # Usage of predictions from another experiment
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Path leading to predictions of another model')

    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=1010710, help='Seed used during model evaluations')

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
    from hps.warmup_hps import ENET_HPS, ENET_GGE_HPS, GATHPS, GGEHPS, MLP_HPS, RF_HPS, XGBOOST_HPS
    from settings.paths import Paths
    from src.data.processing.datasets import PetaleDataset
    from src.data.processing.gnn_datasets import PetaleKGNNDataset
    from src.data.processing.feature_selection import FeatureSelector
    from src.data.processing.sampling import extract_masks, GeneChoice, get_learning_one_data, push_valid_to_train
    from src.models.blocks.genes_signature_block import GeneEncoder, GeneGraphEncoder, GeneGraphAttentionEncoder
    from src.models.gat import PetaleGATR, GATHP
    from src.models.gge import PetaleGGE
    from src.models.mlp import PetaleMLPR, MLPHP
    from src.models.random_forest import PetaleRFR
    from src.models.xgboost_ import PetaleXGBR
    from src.training.evaluation import Evaluator
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.hyperparameters import Range
    from src.utils.score_metrics import AbsoluteError, Pearson, RootMeanSquaredError, SquaredError

    # Arguments parsing
    args = argument_parser()

    # Initialization of DataManager and sampler
    manager = PetaleDataManager()

    # We extract needed data
    if args.genes_subgroup:
        genes_selection = GeneChoice.SIGNIFICANT
        gene_cols = SIGNIFICANT_CHROM_POS_BMI
    elif args.all_genes:
        genes_selection = GeneChoice.ALL
        gene_cols = ALL_CHROM_POS_BMI
    else:
        genes_selection = None
        gene_cols = None

    genes = True if genes_selection is not None else False
    df, target, cont_cols, cat_cols = get_learning_one_data(manager, genes_selection)

    # Extraction of masks
    masks = extract_masks(Paths.BMI_MASK, k=args.nb_outer_splits, l=args.nb_inner_splits)
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
    eval_id = ""
    if genes:
        if args.all_genes:
            eval_id += "_gen2"
        else:
            eval_id += "_gen1"
    if args.enable_sam:
        eval_id += "_sam"

    # We save the Sharpness-Aware Minimization search space
    sam_search_space = {Range.MIN: 0.05, Range.MAX: 2}

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
                              evaluation_name=f"RandomForest_bmi{eval_id}",
                              hps=RF_HPS,
                              n_trials=200,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
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

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleXGBR,
                              dataset=dataset,
                              masks=masks_without_val,
                              evaluation_name=f"XGBoost_bmi{eval_id}",
                              hps=XGBOOST_HPS,
                              n_trials=200,
                              evaluation_metrics=evaluation_metrics,
                              feature_selector=feature_selector,
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
            return {'max_epochs': 250,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of fixed_params for MLP
        fixed_params = update_fixed_params(dataset)

        # Update of hyperparameters
        if args.enable_sam:
            MLP_HPS[MLPHP.RHO.name] = sam_search_space

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"MLP_bmi{eval_id}",
                              hps=MLP_HPS,
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
            return {'max_epochs': 200,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes}

        # Saving of fixed_params for ENET
        fixed_params = update_fixed_params(dataset)

        # Update of hyperparameters
        if args.enable_sam:
            ENET_HPS[MLPHP.RHO.name] = sam_search_space

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"enet_bmi{eval_id}",
                              hps=ENET_HPS,
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
            return {'max_epochs': 200,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor}


        # Saving of fixed_params for GGE + ENET
        fixed_params = update_fixed_params(dataset)

        # Update of hyperparameters
        if args.enable_sam:
            ENET_GGE_HPS[MLPHP.RHO.name] = sam_search_space

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggeEnet_bmi{eval_id}",
                              hps=ENET_GGE_HPS,
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
            return {'max_epochs': 200,
                    'patience': 50,
                    'num_cont_col': len(dts.cont_idx),
                    'cat_idx': dts.cat_idx,
                    'cat_sizes': dts.cat_sizes,
                    'cat_emb_sizes': dts.cat_sizes,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'gene_encoder_constructor': gene_encoder_constructor}


        # Saving of fixed_params for GGAE + ENET
        fixed_params = update_fixed_params(dataset)

        # Update of hyperparameters
        if args.enable_sam:
            ENET_GGE_HPS[MLPHP.RHO.name] = sam_search_space

        # Creation of evaluator
        evaluator = Evaluator(model_constructor=PetaleMLPR,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggaeEnet_bmi{eval_id}",
                              hps=ENET_GGE_HPS,
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

        print("Time Taken for GGAE (minutes): ", round((time.time() - start) / 60, 2))

    """
    GAT experiment
    """
    if args.gat and args.baselines:

        # Start timer
        start = time.time()

        # Creation of the dataset
        if args.conditional_column:
            cond_cat_col = SEX
            nb_neighbor = 5
        else:
            cond_cat_col = None
            nb_neighbor = 7

        GAT_options = [("", False)] if not args.weighted_similarity else [("", False), ("w", True)]

        for prefix, w_sim in GAT_options:

            dataset = PetaleKGNNDataset(df, target, k=nb_neighbor, similarity=PetaleKGNNDataset.COSINE,
                                        weighted_similarity=w_sim,
                                        cont_cols=cont_cols, cat_cols=cat_cols,
                                        conditional_cat_col=cond_cat_col, classification=False)

            # Creation of function to update fixed params
            def update_fixed_params(dts):
                return {'num_cont_col': len(dts.cont_idx),
                        'cat_idx': dts.cat_idx,
                        'cat_sizes': dts.cat_sizes,
                        'cat_emb_sizes': dts.cat_sizes,
                        'max_epochs': 200,
                        'patience': 50}

            # Saving of original fixed params for HAN
            fixed_params = update_fixed_params(dataset)

            # Update of hyperparameters
            if args.enable_sam:
                GATHPS[GATHP.RHO.name] = sam_search_space

            # Creation of the evaluator
            evaluator = Evaluator(model_constructor=PetaleGATR,
                                  dataset=dataset,
                                  masks=masks,
                                  evaluation_name=f"{prefix}GAT_bmi{eval_id}",
                                  hps=GATHPS,
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
            return {'max_epochs': 200,
                    'patience': 50,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'hidden_size': 3,
                    'signature_size': args.signature_size,
                    'genes_emb_sharing': args.embedding_sharing,
                    'aggregation_method': 'att'}

        # Saving of original fixed params for GGAE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleGGE,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"ggae_bmi{eval_id}",
                              hps=GGEHPS,
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
            return {'max_epochs': 200,
                    'patience': 50,
                    'gene_idx_groups': dts.gene_idx_groups,
                    'hidden_size': 3,
                    'signature_size': args.signature_size,
                    'genes_emb_sharing': args.embedding_sharing,
                    'aggregation_method': 'avg'}

        # Saving of original fixed params for GGAE
        fixed_params = update_fixed_params(dataset)

        # Creation of the evaluator
        evaluator = Evaluator(model_constructor=PetaleGGE,
                              dataset=dataset,
                              masks=masks,
                              evaluation_name=f"gge_bmi{eval_id}",
                              hps=GGEHPS,
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

        print("Time Taken for Self Supervised GGE (minutes): ", round((time.time() - start) / 60, 2))

    print("Overall time (minutes): ", round((time.time() - first_start) / 60, 2))