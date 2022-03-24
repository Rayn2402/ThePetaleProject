"""
Filename: L1_REF_RAW.py

Authors: Nicolas Raymond

Description: This file stores the procedure to execute in order to obtain
             "L1_REF_RAW" table in the database.

         L1_REF_RAW contains :

            Features:
             - SEX
             - AGE AT DIAGNOSIS
             - DURATION OF TREATMENT (DT)
             - RADIOTHERAPY DOSE (0; >0)
             - DOX DOSE
             - DEX (0; >0, <=Med; >Med) where Med is the median without 0's
             - GESTATIONAL AGE AT BIRTH (<37w, >=37w)
             - WEIGHT AT BIRTH (<2500g, >=2500g)
             - METHO DOSE
             - CORTICO DOSE

            Target:
            - EF (ejection fraction)

Date of last modification : 2022/03/22
"""

import pandas as pd
import sys

from numpy import zeros
from os.path import dirname, join, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
    from settings.paths import Paths
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.extraction.helpers import get_missing_update
    from src.data.processing.cleaning import DataCleaner

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager()

    # We build a data cleaner
    data_cleaner = DataCleaner(join(Paths.CLEANING_RECORDS, "REF"), column_thresh=COLUMN_REMOVAL_THRESHOLD,
                               row_thresh=ROW_REMOVAL_THRESHOLD, outlier_alpha=OUTLIER_ALPHA,
                               min_n_per_cat=MIN_N_PER_CAT, max_cat_percentage=MAX_CAT_PERCENTAGE)

    # We save the variables needed from BASELINE_FEATURES_AND_COMPLICATIONS
    baseline_vars = [PARTICIPANT, SEX, AGE_AT_DIAGNOSIS, DT, RADIOTHERAPY_DOSE,
                     DOX, DEX, BIRTH_AGE, BIRTH_WEIGHT]

    # We save the variables needed from CARDIO2
    cardio2_vars = [PARTICIPANT, TAG, EF]

    # We retrieve the tables needed
    baseline_df = data_manager.get_table(BASE_FEATURES_AND_COMPLICATIONS, baseline_vars)
    cardio2_df = data_manager.get_table(CARDIO_2, cardio2_vars)
    chrom_pos_df = data_manager.get_table(ALLGENES)
    metho_cortico_df = data_manager.get_table(METHO_CORTICO_TABLE)

    # We only keep survivors from phase 1
    cardio2_df = cardio2_df[cardio2_df[TAG] == PHASE]
    cardio2_df.drop([TAG], axis=1, inplace=True)

    # We proceed to table concatenation (baselines + ef)
    intermediate_df = pd.merge(baseline_df, cardio2_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(intermediate_df[PARTICIPANT].values) if p not in list(baseline_df[PARTICIPANT].values)]
    print(f"Participant with missing ejection fraction: {removed}")
    print(f"Total : {len(removed)}")

    # We create the REDUCED_EF_TARGET table
    cardio2_df[REDUCED_EF] = zeros(cardio2_df.shape[0])
    cardio2_df.loc[(cardio2_df[EF] < 55), [REDUCED_EF]] = 1
    cardio2_df.drop([EF], axis=1, inplace=True)
    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(cardio2_df.columns)}
    data_manager.create_and_fill_table(cardio2_df, REDUCED_EF_TARGET, types, primary_key=[PARTICIPANT])

    # We proceed to table concatenation (baselines + ef + metho cortico)
    intermediate_df2 = pd.merge(intermediate_df, metho_cortico_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(intermediate_df[PARTICIPANT].values) if p not in list(intermediate_df2[PARTICIPANT].values)]
    print(f"Participant with missing metho-cortico data: {removed}")
    print(f"Total : {len(removed)}")

    # We proceed to table concatenation (baselines + ef + genes)
    complete_df = pd.merge(intermediate_df2, chrom_pos_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(intermediate_df2[PARTICIPANT].values) if p not in list(complete_df[PARTICIPANT].values)]
    print(f"Missing participant from ALL GENES: {removed}")
    print(f"Total : {len(removed)}")

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We remove rows and columns with too many missing values and stores other cleaning suggestions
    complete_df = data_cleaner(complete_df)

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We create the dictionary needed to create the table
    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(complete_df.columns)}

    # We make sure that the target is at the end
    types.pop(EF)
    types[EF] = TYPES[EF]

    # We create the RAW learning table
    data_manager.create_and_fill_table(complete_df, f"{LEARNING_2}_{RAW}", types, primary_key=[PARTICIPANT])
