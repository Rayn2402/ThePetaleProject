"""
Filename: paths.py

Authors : Nicolas Raymond

Description : Stores a custom enumeration of the important paths within the project

Date of last modification: 2021/10/19
"""

from os.path import dirname, join


class Paths:
    """
    Paths of important directories and files
    """
    PROJECT_DIR: str = dirname(dirname(__file__))
    CHECKPOINTS: str = join(PROJECT_DIR, "checkpoints")
    CSV_FILES: str = join(PROJECT_DIR, "csv")
    HYPERPARAMETERS: str = join(PROJECT_DIR, "hps")
    MASKS: str = join(PROJECT_DIR, "masks")
    WARMUP_MASK: str = join(MASKS, "warmup_mask.json")
    WARMUP_HOLDOUT_MASK: str = join(MASKS, "warmup_holdout_mask.json")
    OBESITY_MASK: str = join(MASKS, "obesity_mask.json")
    OBESITY_HOLDOUT_MASK: str = join(MASKS, "obesity_holdout_mask.json")
    REF_MASK: str = join(MASKS, "ref_mask.json")
    MODELS: str = join(PROJECT_DIR, "models")
    RECORDS: str = join(PROJECT_DIR, "records")
    CLEANING_RECORDS: str = join(RECORDS, "cleaning")
    DESC_RECORDS: str = join(RECORDS, "descriptive_analyses")
    DESC_CHARTS: str = join(DESC_RECORDS, "charts")
    DESC_STATS: str = join(DESC_RECORDS, "stats")
    EXPERIMENTS_RECORDS: str = join(RECORDS, "experiments")
    FIGURES_RECORDS: str = join(RECORDS, "figures")
    TUNING_RECORDS: str = join(RECORDS, "tuning")
    EXPERIMENTS_SCRIPTS: str = join(PROJECT_DIR, "experiments")
    WARMUP_EXPERIMENTS_SCRIPTS: str = join(EXPERIMENTS_SCRIPTS, "warmup")
    L1_EXPERIMENT_SCRIPTS: str = join(EXPERIMENTS_SCRIPTS, "learning_01")
    SANITY_CHECKS: str = join(PROJECT_DIR, "sanity_checks")

