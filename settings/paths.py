"""
Authors : Nicolas Raymond

Files that contains Paths enumerations

"""
from os.path import dirname, join


class Paths:
    """
    Paths of important folder and files
    """
    PROJECT_DIR: str = dirname(dirname(__file__))
    CHECKPOINTS: str = join(PROJECT_DIR, "checkpoints")
    CSV_FILES: str = join(PROJECT_DIR, "csv")
    HYPERPARAMETERS: str = join(PROJECT_DIR, "hps")
    MASKS: str = join(PROJECT_DIR, "masks")
    RECORDS: str = join(PROJECT_DIR, "records")
    CLEANING_RECORDS: str = join(RECORDS, "cleaning")
    DESC_RECORDS: str = join(RECORDS, "descriptive_analyses")
    DESC_CHARTS: str = join(DESC_RECORDS, "charts")
    DESC_STATS: str = join(DESC_RECORDS, "stats")
    EXPERIMENTS_RECORDS: str = join(RECORDS, "experiments")
    TUNING_RECORDS: str = join(RECORDS, "tuning")
    EXPERIMENTS_SCRIPTS: str = join(PROJECT_DIR, "experiments")
    WARMUP_EXPERIMENTS_SCRIPTS: str = join(EXPERIMENTS_SCRIPTS, "warm_up")
    SANITY_CHECKS: str = join(PROJECT_DIR, "sanity_checks")

