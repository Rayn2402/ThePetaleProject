"""
Author : Nicolas Raymond

This file is used to store helpful constants for table extracting

"""


# TABLE NAMES
GEN_1 = "General_1_Demographic Questionnaire"
GEN_2 = "General_2_CRF Hematology-Oncology"
CARDIO_0 = "Cardio_0_Évaluation à l'Effort (EE)"
CARDIO_3 = "Cardio_3_Questionnaire d'Activité Physique (QAP)"
CARDIO_4 = "Cardio_4_Test de Marche de 6 Minutes (TDM6)"
ID_TABLE = "VO2_ID"
DEX_DOX_TABLE = "DEX_DOX"
LEARNING_0 = "Learning_0_WARMUP"
LEARNING_1 = "Learning_1_6MWT"
LEARNING_2 = "Learning_2_EE"
LEARNING_3 = "Learning_3_6MWT_AND_GENES"
LEARNING_4 = "Learning_4_EE_AND_GENES"

# TABLE KEYS AND FILTERS
PARTICIPANT, TAG, DATE = "Participant", "Tag", "Date"
PHASE = "Phase 1"
INNER = "inner"
PKEY = [PARTICIPANT, TAG]

# DATA TYPE
DATE_TYPE = "date"
NUMERIC_TYPE = "numeric"
CATEGORICAL_TYPE = "text"

# GENERAL 1 COLUMNS
SEX = "34500 Sex"
DATE_OF_BIRTH = "34501 Date of birth (survivor)"
HEIGHT = "34502 Height"
WEIGHT = "34503 Weight"
SMOKING = "34604 Is currently smoking?"

# GENERAL 2 COLUMNS
DATE_OF_DIAGNOSIS = "34471 Date of diagnosis"
AGE_AT_DIAGNOSIS = "34472 Age at diagnosis"
DATE_OF_TREATMENT_END = "34474 Date of treatment end"
RADIOTHERAPY = "34479 Radiotherapy?"
RADIOTHERAPY_DOSE = "34480 Radiotherapy dose"

# CARDIO 0 COLUMNS
VO2_MAX = "35006 EE_VO2_max"
VO2_MAX_PRED = "35008 EE_VO2_max_pred"
VO2R_MAX = "35009 EE_VO2r_max"
TAS_REST = "35020 EE_TAS_rest"
TAD_REST = "35021 EE_TAD_rest"

# CARDIO 3 COLUMNS
QAPL8 = "35116 QAPL8"

# CARDIO 4 COLUMNS
TDM6_DIST = "35142 TDM6_Distance_2"
TDM6_HR_REST = "35143 TDM6_HR_rest_2"
TDM6_HR_END = "35149 TDM6_HR_6_2"
TDM6_TAS_END = "35152 TDM6_TAS_effort_2"
TDM6_TAD_END = "35153 TDM6_TAD_effort_2"

# DEX_DOX COLUMNS
DEX = 'DEX (mg/m2)'
DOX = 'DOX (mg/m2)'

# NEW COLUMNS NAMES
TSEOT = "Time since end of treatment"
DT = "Duration of treatment"
AGE = "Age"
MVLPA = "MVLPA"
FITNESS = "Fitness"
FITNESS_LVL = "Fitness lvl"
DEX_PRESENCE = "DEX used in treatment"

# TYPE DICT
TYPES = {PARTICIPANT: CATEGORICAL_TYPE,
         TAG: CATEGORICAL_TYPE,
         DATE: DATE_TYPE,
         SEX: CATEGORICAL_TYPE,
         DATE_OF_BIRTH: DATE_TYPE,
         HEIGHT: NUMERIC_TYPE,
         WEIGHT: NUMERIC_TYPE,
         SMOKING: CATEGORICAL_TYPE,
         DATE_OF_DIAGNOSIS: DATE_TYPE,
         AGE_AT_DIAGNOSIS: NUMERIC_TYPE,
         DATE_OF_TREATMENT_END: DATE_TYPE,
         RADIOTHERAPY: CATEGORICAL_TYPE,
         RADIOTHERAPY_DOSE: NUMERIC_TYPE,
         VO2_MAX: NUMERIC_TYPE,
         VO2_MAX_PRED: NUMERIC_TYPE,
         VO2R_MAX: NUMERIC_TYPE,
         TAS_REST: NUMERIC_TYPE,
         TAD_REST: NUMERIC_TYPE,
         QAPL8: NUMERIC_TYPE,
         TDM6_DIST: NUMERIC_TYPE,
         TDM6_HR_REST: NUMERIC_TYPE,
         TDM6_HR_END: NUMERIC_TYPE,
         TDM6_TAS_END: NUMERIC_TYPE,
         TDM6_TAD_END: NUMERIC_TYPE,
         DEX: NUMERIC_TYPE,
         DOX: NUMERIC_TYPE,
         TSEOT: NUMERIC_TYPE,
         DT: NUMERIC_TYPE,
         AGE: NUMERIC_TYPE,
         MVLPA: NUMERIC_TYPE,
         FITNESS_LVL: NUMERIC_TYPE,
         DEX_PRESENCE: CATEGORICAL_TYPE
         }




