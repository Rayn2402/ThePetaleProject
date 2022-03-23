"""
Filename: L1_OBESITY_RAW.py

Authors: Nicolas Raymond

Description: This file stores the procedure to execute in order to obtain
             "L1_OBESITY_RAW" table in the database.

         L1_OBESITY_RAW contains :

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
            - OBESITY

Date of last modification : 2022/03/15
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
    from src.data.extraction.helpers import get_abs_years_timelapse, get_missing_update
    from src.data.processing.cleaning import DataCleaner
    from src.utils.visualization import visualize_class_distribution

    # We build a PetaleDataManager that will help interacting with PETALE database
    data_manager = PetaleDataManager()

    # We build a data cleaner
    data_cleaner = DataCleaner(join(Paths.CLEANING_RECORDS, "OBESITY"), column_thresh=COLUMN_REMOVAL_THRESHOLD,
                               row_thresh=ROW_REMOVAL_THRESHOLD, outlier_alpha=OUTLIER_ALPHA,
                               min_n_per_cat=MIN_N_PER_CAT, max_cat_percentage=MAX_CAT_PERCENTAGE)

    # We save the variables needed from BASELINE_FEATURES_AND_COMPLICATIONS
    baseline_vars = [PARTICIPANT, SEX, AGE_AT_DIAGNOSIS, DT, RADIOTHERAPY_DOSE,
                     DOX, DEX, BIRTH_AGE, BIRTH_WEIGHT]

    # We save the variables needed from GEN_1 to calculate the age
    gen_1_vars = [PARTICIPANT, TAG, DATE, DATE_OF_BIRTH]

    # We retrieve the tables needed
    baseline_df = data_manager.get_table(BASE_FEATURES_AND_COMPLICATIONS, baseline_vars)
    age_df = data_manager.get_table(GEN_1, gen_1_vars)
    chrom_pos_df = data_manager.get_table(ALLGENES)
    metho_cortico_df = data_manager.get_table(METHO_CORTICO_TABLE)

    # We only keep survivors from phase 1 and calculate their ages
    age_df = age_df[age_df[TAG] == PHASE]
    age_df = age_df[~(age_df[DATE].isnull() | age_df[DATE_OF_BIRTH].isnull())]
    get_abs_years_timelapse(df=age_df, new_col=AGE, first_date=DATE_OF_BIRTH, second_date=DATE)
    age_df.drop([DATE, DATE_OF_BIRTH, TAG], axis=1, inplace=True)

    # We proceed to table concatenation (baselines + ages)
    intermediate_df = pd.merge(age_df, baseline_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(baseline_df[PARTICIPANT].values) if p not in list(intermediate_df[PARTICIPANT].values)]
    print(f"Participant with missing ages: {removed}")
    print(f"Total : {len(removed)}")

    # We create the dataframe with the total body fat
    body_fat_df = pd.read_csv(join(Paths.CSV_FILES, "total_body_fat.csv"), sep=';')
    body_fat_df[PARTICIPANT] = body_fat_df[PARTICIPANT].astype('string').apply(data_manager.fill_participant_id)
    body_fat_df[TOTAL_BODY_FAT] = body_fat_df[TOTAL_BODY_FAT].astype('float')
    body_fat_df = body_fat_df[~(body_fat_df[TOTAL_BODY_FAT].isnull())]

    # We proceed to table concatenation (baselines + ages + body fat)
    intermediate_df2 = pd.merge(intermediate_df, body_fat_df, on=[PARTICIPANT], how=INNER)
    intermediate_df2.drop([AGE], axis=1, inplace=True)
    removed = [p for p in list(intermediate_df[PARTICIPANT].values) if p not in list(intermediate_df2[PARTICIPANT].values)]
    print(f"Participant with missing body fat: {removed}")
    print(f"Total : {len(removed)}")

    # We proceed to table concatenation (baselines + ages + body fat + metho cortico)
    intermediate_df3 = pd.merge(intermediate_df2, metho_cortico_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(intermediate_df2[PARTICIPANT].values) if p not in list(intermediate_df3[PARTICIPANT].values)]
    print(f"Participant with missing metho-cortico data: {removed}")
    print(f"Total : {len(removed)}")

    # We create the obesity table
    obesity_df = pd.merge(intermediate_df, body_fat_df, on=[PARTICIPANT], how=INNER)
    percentile = obesity_df.loc[obesity_df[AGE] < 18, [TOTAL_BODY_FAT]].quantile(0.95).to_numpy().item()
    obesity_df[OBESITY] = zeros(obesity_df.shape[0])
    obesity_df.loc[(obesity_df[SEX] == 'Women') & (obesity_df[AGE] > 18) & (obesity_df[TOTAL_BODY_FAT] > 35), [OBESITY]] = 1
    obesity_df.loc[(obesity_df[SEX] == 'Men') & (obesity_df[AGE] > 18) & (obesity_df[TOTAL_BODY_FAT] > 25), [OBESITY]] = 1
    obesity_df.loc[(obesity_df[AGE] < 18) & (obesity_df[TOTAL_BODY_FAT] >= percentile), [OBESITY]] = 1
    obesity_df = obesity_df[[PARTICIPANT, AGE, SEX, OBESITY]]

    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(obesity_df.columns)}
    data_manager.create_and_fill_table(obesity_df, OBESITY_TARGET, types, primary_key=[PARTICIPANT])

    # We proceed to table concatenation (baselines + ages + body fat + genes)
    complete_df = pd.merge(intermediate_df3, chrom_pos_df, on=[PARTICIPANT], how=INNER)
    removed = [p for p in list(intermediate_df3[PARTICIPANT].values) if p not in list(complete_df[PARTICIPANT].values)]
    print(f"Missing participant from ALL GENES: {removed}")
    print(f"Total : {len(removed)}")

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We remove rows and columns with too many missing values and stores other cleaning suggestions
    complete_df = data_cleaner(complete_df)

    # We create a dummy column that combines sex and VO2 quartiles
    complete_df[WARMUP_DUMMY] = pd.qcut(complete_df[TOTAL_BODY_FAT].astype(float).values, 2, labels=False)
    complete_df[WARMUP_DUMMY] = complete_df[SEX] + complete_df[WARMUP_DUMMY].astype(str)
    complete_df[WARMUP_DUMMY] = complete_df[WARMUP_DUMMY].apply(func=lambda x: WARMUP_DUMMY_DICT_INT[x])
    visualize_class_distribution(complete_df[WARMUP_DUMMY].values, WARMUP_DUMMY_DICT_NAME)

    # We look at the missing data
    print(f"n_cols : {len(complete_df.columns)}")
    get_missing_update(complete_df)

    # We create the dictionary needed to create the table
    types = {c: TYPES.get(c, CATEGORICAL_TYPE) for c in list(complete_df.columns)}

    # We make sure that the target is at the end
    types.pop(TOTAL_BODY_FAT)
    types[TOTAL_BODY_FAT] = TYPES[TOTAL_BODY_FAT]

    # We create the RAW learning table
    data_manager.create_and_fill_table(complete_df, f"{LEARNING_1}_{RAW}", types, primary_key=[PARTICIPANT])
