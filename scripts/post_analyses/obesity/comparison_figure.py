"""
Filename: comparison_figure.py

Author: Nicolas Raymond

Description: This file is used to create figures that compare
             obesity models on the holdout set

Date of last modification: 2022/07/12
"""

import matplotlib.pyplot as plt
import sys

from os.path import dirname, join, realpath
from pandas import read_csv

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
    from settings.paths import Paths
    from src.data.extraction.constants import AGE, PARTICIPANT, SEX, OBESITY_TARGET
    from src.data.extraction.data_management import PetaleDataManager

    # We read the data
    df = read_csv(join(Paths.CSV_FILES, "obesity_predictions.csv"), index_col=0)

    # We load the data that contains the sex and the age of the participants
    m = PetaleDataManager()
    age_sex_df = m.get_table(OBESITY_TARGET)
    age_sex_df.set_index(PARTICIPANT, inplace=True)
    age_sex_df = age_sex_df[[SEX, AGE]]

    # We merge the sex data to the predictions df
    df = df.join(age_sex_df)

    # Enable LaTeX
    plt.rc('text', usetex=True)

    # Figure creation
    men_df, women_df = df.loc[(df[SEX] == 'Men') & (df[AGE] >= 18)], df.loc[(df[SEX] == 'Women') & (df[AGE] >= 18)]
    children_df = df.loc[df[AGE] < 18]

    plt.axhline(y=35, linestyle='--', color='grey', lw=0.8)
    plt.text(-1, 35.5, 'Women cutoff', fontdict={'size': 6})
    plt.axhline(y=25, linestyle='--', color='grey', lw=0.8)
    plt.text(-1, 25.5, 'Men cutoff', fontdict={'size': 6})
    plt.plot(df.index, df['target'], 'x', color='grey', label='Targets')
    plt.plot(men_df.index, men_df['enetB'], 'd', label='Lin. Reg. (w/o SNPs) (men)', color='#1f77b4')
    plt.plot(women_df.index, women_df['enetB'], 'o', label='Lin. Reg. (w/o SNPs) (women)', color='#1f77b4')
    plt.plot(children_df.index, children_df['enetB'], 's', label='Lin. Reg. (w/o SNPs) (children)', color='#1f77b4')
    plt.plot(men_df.index, men_df['GGAE'], 'd', label='Lin. Reg. + GGAE (men)', color='orange')
    plt.plot(women_df.index, women_df['GGAE'], 'o', label='Lin. Reg. + GGAE (women)', color='orange')
    plt.plot(children_df.index, children_df['GGAE'], 's', label='Lin. Reg. + GGAE (children)', color='orange')
    plt.xticks(rotation=90)

    plt.ylabel('Total Body Fat (\\%)')
    plt.xlabel('Survivors in the holdout set')
    plt.legend(prop={'size': 7})
    plt.tight_layout()

    for f in ['pdf', 'svg']:
        plt.savefig(f'obesity_predictions.{f}')
