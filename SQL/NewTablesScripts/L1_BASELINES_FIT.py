"""
Authors : Nicolas Raymond

This file contains the procedure to execute in order to obtain
"L1_BASELINES_CARDIOMETABOLIC" and "L1_BASELINES_CARDIOMETABOLIC_HOLDOUT" tables.

"""

from SQL.DataManagement.Utils import initialize_petale_data_manager
from Data.Sampling import split_train_test
from SQL.NewTablesScripts.constants import *
from SQL.DataManagement.Helpers import get_missing_update
import pandas as pd


