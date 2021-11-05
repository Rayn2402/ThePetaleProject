"""
Filename: L0_WARMUP_GENES.py

Authors: Nicolas Raymond

Description: This file contains the procedure to execute in order
             to obtain "L0_WARMUP_GENES_RAW".

Date of last modification : 2021/11/05
"""
import pandas as pd
import sys

from os.path import join, realpath, dirname

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.extraction.helpers import get_missing_update
    from src.data.processing.cleaning import DataCleaner
    from src.utils.visualization import visualize_class_distribution

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager()

    # We build a data cleaner
    data_cleaner = DataCleaner(join(Paths.CLEANING_RECORDS, "WARMUP_GENES"), column_thresh=COLUMN_REMOVAL_THRESHOLD,
                               row_thresh=ROW_REMOVAL_THRESHOLD, outlier_alpha=OUTLIER_ALPHA,
                               min_n_per_cat=MIN_N_PER_CAT, max_cat_percentage=MAX_CAT_PERCENTAGE)

    # We save the variables needed from GENERALS
    GEN_vars = [PARTICIPANT, SEX, AGE, WEIGHT, DT, MVLPA, VO2R_MAX]

    # We save the variables needed from 6MWT
    SIXMWT_vars = [PARTICIPANT, TDM6_HR_END, TDM6_DIST]

    # We save a set with all the variables
    all_vars = set(GEN_vars+SIXMWT_vars)

    # We retrieve the tables needed
    gen_df = data_manager.get_table(GENERALS, GEN_vars)
    six_df = data_manager.get_table(SIXMWT, SIXMWT_vars)
    chrom_pos_df = data_manager.get_table(ALLGENES)

    # We remove survivors with missing VO2R_MAX values
    intermediate_df = gen_df[~(gen_df[VO2R_MAX].isnull())]
    removed = [p for p in list(gen_df[PARTICIPANT].values) if p not in list(intermediate_df[PARTICIPANT].values)]
    print(f"Participant with missing VO2: {removed}")
    print(f"Total : {len(removed)}")

    # We proceed to table concatenation
    intermediate_df = pd.merge(intermediate_df, six_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(gen_df[PARTICIPANT].values) if p not in list(intermediate_df[PARTICIPANT].values)]
    print(f"Missing participant from 6MWT: {removed}")
    print(f"Total : {len(removed)}")

    complete_df = pd.merge(intermediate_df, chrom_pos_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(intermediate_df[PARTICIPANT].values) if p not in list(complete_df[PARTICIPANT].values)]
    print(f"Missing participant from ALL GENES: {removed}")
    print(f"Total : {len(removed)}")

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We remove rows and columns with too many missing values and stores other cleaning suggestions
    complete_df = data_cleaner(complete_df)

    # We create a dummy column that combines sex and VO2 quartiles
    complete_df[WARMUP_DUMMY] = pd.qcut(complete_df[VO2R_MAX].astype(float).values, 2, labels=False)
    complete_df[WARMUP_DUMMY] = complete_df[SEX] + complete_df[WARMUP_DUMMY].astype(str)
    complete_df[WARMUP_DUMMY] = complete_df[WARMUP_DUMMY].apply(func=lambda x: WARMUP_DUMMY_DICT_INT[x])
    # visualize_class_distribution(complete_df[WARMUP_DUMMY].values, WARMUP_DUMMY_DICT_NAME)

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We create the dictionary needed to create the table
    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(complete_df.columns)}

    # We make sure that the target is at the end
    types.pop(VO2R_MAX)
    types[VO2R_MAX] = TYPES[VO2R_MAX]

    # We create the RAW learning table
    data_manager.create_and_fill_table(complete_df, f"{LEARNING_0_GENES}_{RAW}", types, primary_key=[PARTICIPANT])
