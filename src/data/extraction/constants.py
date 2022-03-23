"""
Filename: constants.py

Author : Nicolas Raymond

Description: This file is used to store helpful constants for table extraction

Date of last modification : 2021/12/01
"""
from numpy import nan

# SEED VALUE FOR HOLDOUT SETS
SEED = 110796

# DATA CLEANING THRESHOLDS
COLUMN_REMOVAL_THRESHOLD = 0.20
ROW_REMOVAL_THRESHOLD = 0.20
OUTLIER_ALPHA = 1.5
MIN_N_PER_CAT = 6
MAX_CAT_PERCENTAGE = 0.95

# DATA SAMPLING TRESHOLDS
SAMPLING_OUTLIER_ALPHA = 6

# CHILDHOOD OBESITY 95 percentile
OBESITY_PERCENTILE = 42.695

# TABLE NAMES
GEN_1 = "General_1_Demographic Questionnaire"
GEN_2 = "General_2_CRF Hematology-Oncology"
CARDIO_0 = "Cardio_0_Évaluation à l'Effort (EE)"
CARDIO_2 = "Cardio_2_Echocardiography"
CARDIO_3 = "Cardio_3_Questionnaire d'Activité Physique (QAP)"
CARDIO_4 = "Cardio_4_Test de Marche de 6 Minutes (TDM6)"
VO2_ID_TABLE = "VO2_ID"
INVALID_ID_TABLE = "INVALID_ID"
DEX_DOX_TABLE = "DEX_DOX"
METHO_CORTICO_TABLE = "METHO_CORTICO"
PETALE_PANDORA = "PETALE_PANDORA"
SNPS_RARE = "SNPS_RARE"
SNPS_COMMON = "SNPS_COMMON"
SIGNIFICANT_COMMON_SNPS_ID = "SIGNIFICANT_COMMON_SNPS_ID"
SIGNIFICANT_RARE_SNPS_ID = "SIGNIFICANT_RARE_SNPS_ID"
GENERALS = "GENERALS"
TOP5_SNPS_ID = "TOP5_SNPS_ID"
TOP12_SNPS_ID = "TOP12_SNPS_ID"
GENES_5 = "GENES_5"
GENES_12 = "GENES_12"
ALLGENES = "ALL_GENES"
SIXMWT = "6MWT"
BASE_FEATURES_AND_COMPLICATIONS = "BASELINE_FEATURES_AND_COMPLICATIONS"
BASE_FEATURES_AND_COMPLICATIONS_PLUS_FITNESS = "BASE_FEATURES_AND_COMPLICATIONS_+_FITNESS"
LEARNING_0_GENES = "L0_WARMUP_GENES"
LEARNING_0_GENES_HOLDOUT = "L0_WARMUP_GENES_HOLDOUT"
LEARNING_1 = "L1_OBESITY"
LEARNING_1_HOLDOUT = "L1_OBESITY_HOLDOUT"
LEARNING_2 = "L2_REF"
OBESITY_TARGET = "OBESITY_TARGET"
REDUCED_EF_TARGET = "REDUCED_EF_TARGET"
RAW = "RAW"


# TABLE KEYS AND FILTERS
PARTICIPANT, TAG, DATE = "Participant", "Tag", "Date"
PHASE = "Phase 1"
INNER = "inner"
PKEY = [PARTICIPANT, TAG]

# COMMON COLUMNS AMONG TABLES
FORM = "Form"
STATUS = "Status"
REMARKS = "Remarks"

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
BMI = "34504 BMI"
BIRTH_AGE = "34592 Gestational age at birth"
BIRTH_WEIGHT = "34593 Weight at birth"
SEX_CATEGORIES = {0: "Women", 1: "Men"}
BIRTH_AGE_CATEGORIES = {1: "<37w", 2: ">=37w", 9: nan}
BIRTH_WEIGHT_CATEGORIES = {1: "<2500g", 2: ">=2500g", 9: nan}

# GENERAL 2 COLUMNS
DATE_OF_DIAGNOSIS = "34471 Date of diagnosis"
AGE_AT_DIAGNOSIS = "34472 Age at diagnosis"
DATE_OF_TREATMENT_END = "34474 Date of treatment end"
RADIOTHERAPY = "34479 Radiotherapy?"
RADIOTHERAPY_DOSE = "34480 Radiotherapy dose"
METABOLIC_COMPLICATIONS = "34482 Metabolic complications?"
BONE_COMPLICATIONS = "34483 Skeletal/bone complications?"
CARDIAC_COMPLICATIONS = "34484 Cardiac complications?"
NEUROCOGNITIVE_COMPLICATIONS = "34485 Neurocognitive complications?"

# CARDIO 0 COLUMNS
VO2_MAX = "35006 EE_VO2_max"
VO2_MAX_PRED = "35008 EE_VO2_max_pred"
VO2R_MAX = "35009 EE_VO2r_max"
TAS_REST = "35020 EE_TAS_rest"
TAD_REST = "35021 EE_TAD_rest"

# CARDIO 2 COLUMNS
EF = "34972 FE ModeM (cardio)"

# CARDIO 3 COLUMNS
MVLPA = "35116 QAPL8"

# CARDIO 4 COLUMNS
TDM6_DIST = "35142 TDM6_Distance_2"
TDM6_HR_REST = "35143 TDM6_HR_rest_2"
TDM6_HR_END = "35149 TDM6_HR_6_2"
TDM6_TAS_END = "35152 TDM6_TAS_effort_2"
TDM6_TAD_END = "35153 TDM6_TAD_effort_2"

# DEX_DOX COLUMNS
DEX = 'DEX (mg/m2)'
DOX = 'DOX (mg/m2)'

# METHO_CORTICO COLUMNS
METHO = 'Methotrexate'
CORTICO = 'Effective corticosteroid cumulative dose (mg/m2)'

# SIGNIFICANT SNPs COLUMNS
GENES = "Genes"
SNPS_ID = "SNPs ID"
CHROM = "CHROM"
SNPS_POSITION = "POS"
REF = "REF"
ALT = "ALT"
GENE_REF_GEN = "Generefgene"


# SNPS COLUMNS
SNPS_TYPE = "TYPE"

SIGNIFICANT_CHROM_POS_WARMUP = ['7_45932669', '2_179498042', '2_179444939',
                                '2_179397561', '2_179406191', '2_179436020',
                                '2_179457147', '2_179430997', '2_179458591',
                                '17_37884037', '17_4856580', '16_88713236']

SIGNIFICANT_CHROM_POS_OBESITY = ['2_179644855', '14_96703484', '17_37884037',
                                 '6_26091179', '20_46264888', '17_4856580']

SIGNIFICANT_CHROM_POS_REF = ['2_240946766', '2_179397561', '2_179406191',
                             '2_179457147', '2_179458591', '21_44324365',
                             '17_48712705', '21_37444696', '2_179579093',
                             '2_179582327', '2_179583496', '2_179587130',
                             '10_101473218', '2_179436020', '2_179430997',
                             '14_96703484', '1_230845977', '21_37444120',
                             '2_179644855']

ALL_CHROM_POS_WARMUP = ['1_115231254', '1_151062957', '1_156096387', '1_156099669',
                        '1_161182208', '1_226019633', '1_230845977', '1_46032311',
                        '1_66036441', '1_66075952', '10_101473218', '11_236091',
                        '11_68174189', '12_48272895', '12_48501161', '12_6954864',
                        '13_23824818', '13_95859035', '14_96703484', '15_58838010',
                        '16_16173232', '16_57015091', '16_69745145', '16_88713236',
                        '17_26096597', '17_37884037', '17_4856580', '17_48712705',
                        '17_48712711', '17_48761053', '19_18876309', '19_5892954',
                        '19_5893058', '2_179397561', '2_179406191', '2_179421694',
                        '2_179427536', '2_179430997', '2_179432185', '2_179436020',
                        '2_179444939', '2_179451420', '2_179457147', '2_179458591',
                        '2_179464527', '2_179498042', '2_179545859', '2_179554305',
                        '2_179558366', '2_179575511', '2_179579093', '2_179582327',
                        '2_179582537', '2_179583496', '2_179587130', '2_179606538',
                        '2_179620951', '2_179623758', '2_179629461', '2_179644855',
                        '2_179650408', '2_179659912', '2_240923050', '2_240946766',
                        '20_46264888', '21_37444696', '21_37444697', '21_37518706',
                        '21_44324365', '22_42486723', '22_46614274', '3_119395799',
                        '3_122003769', '3_12393125', '4_120241902', '4_23814707',
                        '4_2906707', '4_2916762', '6_110760008', '6_110763875',
                        '6_12296255', '6_26091179', '6_26093141', '6_29910371',
                        '6_29910663', '6_29910719', '6_29910752', '6_29911218',
                        '6_29911222', '6_29912280', '6_29912297', '6_29912333',
                        '6_29912345', '6_29912348', '6_29912386', '6_29913037',
                        '7_150732812', '7_20687604', '7_20691047', '7_20762646',
                        '7_45932669', '7_94946084', '9_86900369', '9_86917301',
                        '1_160106787', '10_101553324', '10_115674837', '10_115674838',
                        '11_113283484', '12_117725904', '13_95863008', '15_51507968',
                        '16_69748869', '17_37873672', '17_38545824', '18_9122611',
                        '2_179393111', '2_179395560', '2_179396162', '2_179398509',
                        '2_179400895', '2_179403750', '2_179404628', '2_179406294',
                        '2_179408713', '2_179414318', '2_179434516', '2_179438866',
                        '2_179439880', '2_179449131', '2_179542464', '2_179549131',
                        '2_179569387', '2_179571448', '2_179578704', '2_179581835',
                        '2_179582853', '2_179590256', '2_179600648', '2_179604160',
                        '2_179604366', '2_179605725', '2_179611711', '2_179634936',
                        '2_179637861', '2_179641975', '2_179644035', '2_179658175',
                        '6_29910698', '2_31590917', '20_46256424', '21_37444120',
                        '3_119526203', '3_123451932', '6_132189263', '6_29910340',
                        '6_29910602', '6_29910688', '6_29910693', '6_29910721',
                        '6_29910750', '6_29911296', '6_29911306', '6_29912315',
                        '6_29912373', '7_117149147', '7_20682884', '7_87160618']

ALL_CHROM_POS_OBESITY = ['1_115231254', '1_151062957', '1_156096387', '1_156099669',
                         '1_161182208', '1_226019633', '1_230845977', '1_46032311',
                         '1_66036441', '1_66075952', '10_101473218', '11_236091',
                         '11_68174189', '12_48272895', '12_48501161', '12_6954864',
                         '13_23824818', '13_95859035', '14_96703484', '15_58838010',
                         '16_16173232', '16_57015091', '16_69745145', '16_88713236',
                         '17_26096597', '17_37884037', '17_4856580', '17_48712705',
                         '17_48712711', '17_48761053', '19_18876309', '19_5892954',
                         '19_5893058', '2_179397561', '2_179406191', '2_179421694',
                         '2_179427536', '2_179430997', '2_179432185', '2_179436020',
                         '2_179444939', '2_179451420', '2_179457147', '2_179458591',
                         '2_179464527', '2_179498042', '2_179545859', '2_179554305',
                         '2_179558366', '2_179575511', '2_179579093', '2_179582327',
                         '2_179582537', '2_179583496', '2_179587130', '2_179606538',
                         '2_179620951', '2_179623758', '2_179629461', '2_179644855',
                         '2_179650408', '2_179659912', '2_240923050', '2_240946766',
                         '20_46264888', '21_37444696', '21_37444697', '21_37518706',
                         '21_44324365', '22_42486723', '22_46614274', '3_119395799',
                         '3_122003769', '3_12393125', '4_120241902', '4_23814707',
                         '4_2906707', '4_2916762', '6_110760008', '6_110763875',
                         '6_12296255', '6_26091179', '6_26093141', '6_29910371',
                         '6_29910663', '6_29910719', '6_29910752', '6_29912280',
                         '6_29912297', '6_29912333', '6_29912345', '6_29912348',
                         '6_29912386', '6_29913037', '7_150732812', '7_20687604',
                         '7_20691047', '7_20762646', '7_45932669', '7_94946084',
                         '9_86900369', '9_86917301', '1_115222237', '1_156096376',
                         '1_160106787', '10_101553324', '10_115674837', '10_115674838',
                         '11_113283484', '12_117725904', '13_95863008', '15_51507968',
                         '16_69748869', '17_37873672', '17_38545824', '18_9122611',
                         '2_179393111', '2_179395560', '2_179396162', '2_179398509',
                         '2_179400895', '2_179403750', '2_179404628', '2_179406294',
                         '2_179408713', '2_179414318', '2_179434516', '2_179438866',
                         '2_179439880', '2_179449131', '2_179472292', '2_179542464',
                         '2_179581835', '2_179582853', '2_179590256', '2_179600648',
                         '2_179604160', '2_179604366', '2_179605725', '2_179611711',
                         '2_179634936', '2_179637861', '2_179641975', '2_179658175',
                         '6_29910698', '2_31590917', '2_46611678', '20_46256424',
                         '21_37444120', '3_119526203', '4_2883668', '6_132189263',
                         '6_29910340', '6_29910602', '6_29910688', '6_29910693',
                         '6_29910721', '6_29910750', '6_29911296', '6_29911306',
                         '6_29912315', '6_29912373', '7_117149147', '7_20682884',
                         '7_87160618', '7_87179813']

ALL_CHROM_POS_REF = ['1_115231254', '1_151062957', '1_156096387', '1_156099669',
                     '1_161182208', '1_226019633', '1_230845977', '1_46032311',
                     '1_66036441', '1_66075952', '10_101473218', '11_236091',
                     '11_68174189', '12_48272895', '12_48501161', '12_6954864',
                     '13_23824818', '13_95859035', '14_96703484', '15_58838010',
                     '16_16173232', '16_57015091', '16_69745145', '16_88713236',
                     '17_26096597', '17_37884037', '17_4856580', '17_48712705',
                     '17_48712711', '17_48761053', '19_18876309', '19_5892954',
                     '19_5893058', '2_179397561', '2_179406191', '2_179421694',
                     '2_179427536', '2_179430997', '2_179432185', '2_179436020',
                     '2_179444939', '2_179451420', '2_179457147', '2_179458591',
                     '2_179464527', '2_179498042', '2_179545859', '2_179554305',
                     '2_179558366', '2_179575511', '2_179579093', '2_179582327',
                     '2_179582537', '2_179583496', '2_179587130', '2_179606538',
                     '2_179620951', '2_179623758', '2_179629461', '2_179644855',
                     '2_179650408', '2_179659912', '2_240923050', '2_240946766',
                     '20_46264888', '21_37444696', '21_37444697', '21_37518706',
                     '21_44324365', '22_42486723', '22_46614274', '3_119395799',
                     '3_122003769', '3_12393125', '4_120241902', '4_23814707',
                     '4_2906707', '4_2916762', '6_110760008', '6_110763875',
                     '6_12296255', '6_26091179', '6_26093141', '6_29910371',
                     '6_29910663', '6_29910719', '6_29910752', '6_29912280',
                     '6_29912297', '6_29912333', '6_29912345', '6_29912348',
                     '6_29912386', '6_29913037', '7_150732812', '7_20687604',
                     '7_20691047', '7_20762646', '7_45932669', '7_94946084',
                     '9_86900369', '9_86917301', '1_115222237', '1_156096376',
                     '1_160106787', '10_101553324', '10_115674837', '10_115674838',
                     '11_113283484', '12_117725904', '13_95863008', '15_51507968',
                     '16_69748869', '17_37873672', '17_38545824', '18_9122611',
                     '2_179393111', '2_179395560', '2_179396162', '2_179398509',
                     '2_179400895', '2_179403750', '2_179404628', '2_179406294',
                     '2_179408713', '2_179414318', '2_179434516', '2_179438866',
                     '2_179439880', '2_179449131', '2_179472292', '2_179542464',
                     '2_179581835', '2_179582853', '2_179586604', '2_179590256',
                     '2_179600648', '2_179604160', '2_179604366', '2_179605725',
                     '2_179611711', '2_179634936', '2_179634961', '2_179637861',
                     '2_179641975', '2_179658175', '6_29910698', '2_31590917',
                     '2_46611678', '20_46256424', '21_37444120', '3_119526203',
                     '4_2883668', '6_132189263', '6_29910340', '6_29910602',
                     '6_29910688', '6_29910693', '6_29910721', '6_29910750',
                     '6_29911296', '6_29911306', '6_29912315', '6_29912373',
                     '7_117149147', '7_20682884', '7_87160618', '7_87179813']


# PETALE PANDORA COLUMNS
REF_NAME = "Reference name"

# NEW COLUMNS NAMES
TSEOT = "Time since end of treatment"
DT = "Duration of treatment"
AGE = "Age"
FITNESS = "Fitness"
FITNESS_COMPLICATIONS = "Fitness complications?"
CARDIOMETABOLIC_COMPLICATIONS = "Cardiometabolic complications?"
COMPLICATIONS = "Complications?"
TOTAL_BODY_FAT = "Total body fat (%)"
OBESITY = "Obesity"
REDUCED_EF = "Reduced Ejection Fraction"
WARMUP_DUMMY = "Dummy"

# WARMUP DUMMY VARIABLE DICTIONARY
WARMUP_DUMMY_DICT_INT = {"Women0": 0, "Women1": 1, "Men0": 2, "Men1": 3}
WARMUP_DUMMY_DICT_NAME = {0: "Women - q0", 1: "Women - q1", 2: "Men - q0", 3: "Men - q1"}

# TYPE DICT
TYPES = {PARTICIPANT: CATEGORICAL_TYPE,
         TAG: CATEGORICAL_TYPE,
         DATE: DATE_TYPE,
         FORM: CATEGORICAL_TYPE,
         REMARKS: CATEGORICAL_TYPE,
         STATUS: CATEGORICAL_TYPE,
         SEX: CATEGORICAL_TYPE,
         DATE_OF_BIRTH: DATE_TYPE,
         HEIGHT: NUMERIC_TYPE,
         WEIGHT: NUMERIC_TYPE,
         SMOKING: CATEGORICAL_TYPE,
         DATE_OF_DIAGNOSIS: DATE_TYPE,
         AGE_AT_DIAGNOSIS: NUMERIC_TYPE,
         DATE_OF_TREATMENT_END: DATE_TYPE,
         RADIOTHERAPY: CATEGORICAL_TYPE,
         RADIOTHERAPY_DOSE: CATEGORICAL_TYPE,
         VO2_MAX: NUMERIC_TYPE,
         VO2_MAX_PRED: NUMERIC_TYPE,
         VO2R_MAX: NUMERIC_TYPE,
         TAS_REST: NUMERIC_TYPE,
         TAD_REST: NUMERIC_TYPE,
         MVLPA: NUMERIC_TYPE,
         TDM6_DIST: NUMERIC_TYPE,
         TDM6_HR_REST: NUMERIC_TYPE,
         TDM6_HR_END: NUMERIC_TYPE,
         TDM6_TAS_END: NUMERIC_TYPE,
         TDM6_TAD_END: NUMERIC_TYPE,
         DEX: CATEGORICAL_TYPE,
         DOX: NUMERIC_TYPE,
         TSEOT: NUMERIC_TYPE,
         DT: NUMERIC_TYPE,
         AGE: NUMERIC_TYPE,
         FITNESS_COMPLICATIONS: CATEGORICAL_TYPE,
         BIRTH_AGE: CATEGORICAL_TYPE,
         BIRTH_WEIGHT: CATEGORICAL_TYPE,
         METABOLIC_COMPLICATIONS: CATEGORICAL_TYPE,
         BONE_COMPLICATIONS: CATEGORICAL_TYPE,
         CARDIAC_COMPLICATIONS: CATEGORICAL_TYPE,
         NEUROCOGNITIVE_COMPLICATIONS: CATEGORICAL_TYPE,
         CARDIOMETABOLIC_COMPLICATIONS: CATEGORICAL_TYPE,
         COMPLICATIONS: CATEGORICAL_TYPE,
         GENES: CATEGORICAL_TYPE,
         SNPS_ID: CATEGORICAL_TYPE,
         SNPS_TYPE: CATEGORICAL_TYPE,
         CHROM: NUMERIC_TYPE,
         SNPS_POSITION: NUMERIC_TYPE,
         REF_NAME: CATEGORICAL_TYPE,
         WARMUP_DUMMY: NUMERIC_TYPE,
         BMI: NUMERIC_TYPE,
         OBESITY: NUMERIC_TYPE,
         TOTAL_BODY_FAT: NUMERIC_TYPE,
         EF: NUMERIC_TYPE,
         REDUCED_EF: NUMERIC_TYPE,
         METHO: NUMERIC_TYPE,
         CORTICO: NUMERIC_TYPE,
         }

