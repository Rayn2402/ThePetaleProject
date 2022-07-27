"""
Filename: paths.py

Authors : Nicolas Raymond

Description : Stores a custom enumeration of the important paths within the project

Date of last modification: 2021/07/27
"""

from os.path import dirname, join
from src.data.extraction import constants as cst


class Paths:
    """
    Paths of important directories and files
    """
    PROJECT_DIR: str = dirname(dirname(__file__))
    CHECKPOINTS: str = join(PROJECT_DIR, "checkpoints")
    DATA: str = join(PROJECT_DIR, "data")
    VO2_LEARNING_SET_CSV = join(DATA, f"{cst.VO2_LEARNING_SET}.csv")
    VO2_HOLDOUT_SET_CSV = join(DATA, f"{cst.VO2_HOLDOUT_SET}.csv")
    OBESITY_LEARNING_SET_CSV = join(DATA, f"{cst.OBESITY_LEARNING_SET}.csv")
    OBESITY_HOLDOUT_SET_CSV = join(DATA, f"{cst.OBESITY_HOLDOUT_SET}.csv")
    HYPERPARAMETERS: str = join(PROJECT_DIR, "hps")
    MASKS: str = join(PROJECT_DIR, "masks")
    VO2_MASK: str = join(MASKS, "vo2_mask.json")
    VO2_HOLDOUT_MASK: str = join(MASKS, "vo2_holdout_mask.json")
    OBESITY_MASK: str = join(MASKS, "obesity_mask.json")
    OBESITY_HOLDOUT_MASK: str = join(MASKS, "obesity_holdout_mask.json")
    MODELS: str = join(PROJECT_DIR, "models")
    RECORDS: str = join(PROJECT_DIR, "records")
    CLEANING_RECORDS: str = join(RECORDS, "cleaning")
    CSV_FILES: str = join(RECORDS, "csv")
    DESC_RECORDS: str = join(RECORDS, "descriptive_analyses")
    DESC_CHARTS: str = join(DESC_RECORDS, "charts")
    DESC_STATS: str = join(DESC_RECORDS, "stats")
    EXPERIMENTS_RECORDS: str = join(RECORDS, "experiments")
    FIGURES_RECORDS: str = join(RECORDS, "figures")
    TUNING_RECORDS: str = join(RECORDS, "tuning")

