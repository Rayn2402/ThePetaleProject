"""
Filename: descriptive_analyses.py

Authors: Nicolas Raymond

Description: This file is used to generate descriptive analyses of the table
             related to the obesity prediction task.

Date of last modification : 2022/08/08
"""
import sys
from os.path import dirname, realpath

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
    from src.data.extraction.constants import *
    from src.data.extraction.data_management import PetaleDataManager

    # Data manager initialization
    data_manager = PetaleDataManager()

    # Tables extraction
    num_clinical_features = [EOT_BMI, METHO, CORTICO, DOX, DT, AGE_AT_DIAGNOSIS, TOTAL_BODY_FAT]
    cat_clinical_features = [DEX, BIRTH_AGE, RADIOTHERAPY_DOSE]

    snps = sorted(['6_110760008', '17_26096597', '17_37884037', '21_44324365',
                   '1_66036441', '17_4856580', '13_23824818', '17_38545824',
                   '1_161182208', '22_42486723', '6_29912386', '16_88713236',
                   '6_29912280', '21_37518706', '12_48272895', '7_87160618',
                   '1_226019633', '7_45932669', '2_46611678', '6_29912333',
                   '15_58838010', '7_20762646', '17_48712711', '13_95863008',
                   '7_94946084', '4_120241902', '16_69745145', '6_12296255',
                   '2_240946766', '2_179650408'
                   ], key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))

    variables = [PARTICIPANT, SEX] + num_clinical_features + cat_clinical_features + snps
    learning_set = data_manager.get_table(OBESITY_LEARNING_SET, columns=variables)
    holdout_set = data_manager.get_table(OBESITY_HOLDOUT_SET, columns=variables)
    dataset = learning_set.append(holdout_set)

    # We proceed to the descriptive analyses of numerical clinical features
    for table, name in [(learning_set, 'learning_set'), (holdout_set, 'holdout_set'), (dataset, 'dataset')]:
        data_manager.get_table_stats(table[[PARTICIPANT, SEX] + num_clinical_features],
                                     numerical_cols=num_clinical_features, categorical_cols=[],
                                     filename=f'obesity_{name}_num_clin')

    for table, name in [(learning_set, 'learning_set'), (holdout_set, 'holdout_set'), (dataset, 'dataset')]:
        df = data_manager.get_table_stats(table[[PARTICIPANT, SEX] + cat_clinical_features],
                                          numerical_cols=[],
                                          categorical_cols=cat_clinical_features)
        df.drop(0, inplace=True)
        df['Variable Name'] = df['Variable Name'].map(lambda x: x.replace(' ', ''))
        df['Modality'] = df['Variable Name'].map(lambda x: x.split(":")[1])
        df['Variable Name'] = df['Variable Name'].map(lambda x: x.split(":")[0])
        df.sort_values(by=['Variable Name', 'Modality'], inplace=True)
        columns = df.columns.tolist()
        columns.insert(1, "Modality")
        columns.pop()
        df = df[columns]
        data_manager.save_stats_file(f'obesity_{name}_cat_clin', df)

    # We proceed to the descriptive analyses of snps
    for table, name in [(learning_set, 'learning_set'), (holdout_set, 'holdout_set'), (dataset, 'dataset')]:
        df = data_manager.get_table_stats(table[[PARTICIPANT, SEX] + snps], numerical_cols=[], categorical_cols=snps)
        df.drop(0, inplace=True)
        df['Variable Name'] = df['Variable Name'].map(lambda x: x.replace(' ', ''))
        df['Modality'] = df['Variable Name'].map(lambda x: x.split(":")[1])
        df['Variable Name'] = df['Variable Name'].map(lambda x: x.split(":")[0])
        df['Chrom'] = df['Variable Name'].map(lambda x: int(x.split("_")[0]))
        df['Pos'] = df['Variable Name'].map(lambda x: int(x.split("_")[1]))
        df.sort_values(by=['Chrom', 'Pos', 'Modality'], inplace=True)
        df['SNP'] = df['Chrom'].map(str) + ":" + df['Pos'].map(str)
        df.drop(['Variable Name', 'Chrom', 'Pos'], axis=1, inplace=True)
        columns = df.columns.tolist()
        columns.insert(0, "Modality")
        columns.insert(0, "SNP")
        columns.pop()
        columns.pop()
        df = df[columns]
        data_manager.save_stats_file(f'obesity_{name}_snps', df)


