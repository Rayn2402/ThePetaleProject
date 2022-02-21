"""
Filename: L1_BMI_RAW.py

Authors: Nicolas Raymond

Description: This file stores the procedure to execute in order to obtain
             "L1_BMI_RAW" table in the database.

         L1_BMI_RAW contains :

            Features:
             - SEX
             - AGE AT DIAGNOSIS
             - DURATION OF TREATMENT (DT)
             - RADIOTHERAPY DOSE (0; >0)
             - DOX DOSE
             - DEX (0; >0, <=Med; >Med) where Med is the median without 0's
             - GESTATIONAL AGE AT BIRTH (<37w, >=37w)
             - WEIGHT AT BIRTH (<2500g, >=2500g)

            Target:
            - BMI

Date of last modification : 2022/02/01
"""

import pandas as pd
import sys

from os.path import dirname, join, realpath

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
    data_cleaner = DataCleaner(join(Paths.CLEANING_RECORDS, "BMI"), column_thresh=COLUMN_REMOVAL_THRESHOLD,
                               row_thresh=ROW_REMOVAL_THRESHOLD, outlier_alpha=OUTLIER_ALPHA,
                               min_n_per_cat=MIN_N_PER_CAT, max_cat_percentage=MAX_CAT_PERCENTAGE)

    # We save the variables needed from BASELINE_FEATURES_AND_COMPLICATIONS
    baseline_vars = [PARTICIPANT, SEX, AGE_AT_DIAGNOSIS, DT, RADIOTHERAPY_DOSE,
                     DOX, DEX, BIRTH_AGE, BIRTH_WEIGHT]

    # We retrieve the tables needed
    baseline_df = data_manager.get_table(BASE_FEATURES_AND_COMPLICATIONS, baseline_vars)
    bmi_df = data_manager.get_table(GEN_1, [PARTICIPANT, TAG, BMI])
    chrom_pos_df = data_manager.get_table(ALLGENES)

    # We only keep survivors from phase 1 and also remove the ones missing BMI
    bmi_df = bmi_df[bmi_df[TAG] == PHASE]
    intermediate_df = bmi_df[~(bmi_df[BMI].isnull())]
    removed = [p for p in list(bmi_df[PARTICIPANT].values) if p not in list(intermediate_df[PARTICIPANT].values)]
    print(f"Participant with missing BMI: {removed}")
    print(f"Total : {len(removed)}")

    # We proceed to table concatenation
    intermediate_df = pd.merge(intermediate_df, baseline_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(baseline_df[PARTICIPANT].values) if p not in list(intermediate_df[PARTICIPANT].values)]
    print(f"Missing participant from BASELINES: {removed}")
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

    # We create a dummy column that combines sex and BMI quartiles
    complete_df[WARMUP_DUMMY] = pd.qcut(complete_df[BMI].astype(float).values, 2, labels=False)
    complete_df[WARMUP_DUMMY] = complete_df[SEX] + complete_df[WARMUP_DUMMY].astype(str)
    complete_df[WARMUP_DUMMY] = complete_df[WARMUP_DUMMY].apply(func=lambda x: WARMUP_DUMMY_DICT_INT[x])
    visualize_class_distribution(complete_df[WARMUP_DUMMY].values, WARMUP_DUMMY_DICT_NAME)

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We create the dictionary needed to create the table
    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(complete_df.columns)}

    # We make sure that the target is at the end
    types.pop(BMI)
    types[BMI] = TYPES[BMI]

    # We create the RAW learning table
    data_manager.create_and_fill_table(complete_df, f"{LEARNING_1}_{RAW}", types, primary_key=[PARTICIPANT])
