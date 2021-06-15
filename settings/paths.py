"""
Authors : Nicolas Raymond

Files that contains Paths enumerations

"""
import os
from enum import Enum


class Paths(Enum):
    """
    Paths of important folder and files
    """
    PROJECT_DIR: str = os.path.dirname(os.path.dirname(__file__))
    CSV_FILES: str = os.path.join(PROJECT_DIR, "csv")
    HYPERPARAMETERS: str = os.path.join(PROJECT_DIR, "hps")
    MASKS: str = os.path.join(PROJECT_DIR, "masks")
    RECORDS: str = os.path.join(PROJECT_DIR, "records")
    CLEANING_RECORDS: str = os.path.join(RECORDS, "cleaning")
    DESC_RECORDS: str = os.path.join(RECORDS, "descriptive_analyses")
    DESC_CHARTS: str = os.path.join(DESC_RECORDS, "charts")
    DESC_STATS: str = os.path.join(DESC_RECORDS, "stats")

