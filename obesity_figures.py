"""
Filename: obesity_figures.py

Author: Nicolas Raymond

Description: This file is used to create figures that compare obesity model predictions on the holdout set

Date of last modification: 2022/05/30
"""

import matplotlib.pyplot as plt

from os.path import join
from pandas import read_csv
from settings.paths import Paths
from src.data.extraction.constants import PARTICIPANT, SEX
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.sampling import get_learning_one_data

if __name__ == '__main__':

    # We read the data
    df = read_csv(join(Paths.CSV_FILES, "obesity_predictions.csv"), index_col=0)

    # We load the data that contains the sex of the participants
    sex_df, _, _, _, = get_learning_one_data(PetaleDataManager(), genes=None, holdout=True)
    sex_df.set_index(PARTICIPANT, inplace=True)
    sex_df = sex_df[SEX]

    # We merge the sex data to the predictions df
    df = df.join(sex_df)

    # Enable LaTeX
    plt.rc('text', usetex=True)

    # Figure creation
    men_df, women_df = df.loc[df[SEX] == 'Men'], df.loc[df[SEX] == 'Women']

    plt.plot(df.index, df['target'], 'x', color='grey', label='Targets')
    plt.plot(men_df.index, men_df['Enet'], 'd', label='Linear Reg. (without SNPs) (men)', color='#1f77b4')
    plt.plot(women_df.index, women_df['Enet'], 'o', label='Linear Reg. (without SNPs) (women)', color='#1f77b4')
    plt.plot(men_df.index, men_df['GGAE'], 'd', label='Linear Reg. + GGAE (men)', color='orange')
    plt.plot(women_df.index, women_df['GGAE'], 'o', label='Linear Reg. + GGAE (women)', color='orange')
    plt.xticks(rotation=90)

    plt.ylabel('Total Body Fat (\\%)')
    plt.xlabel('Survivors in the holdout set')
    plt.legend()
    plt.tight_layout()

    for f in ['pdf', 'svg']:
        plt.savefig(f'obesity_predictions.{f}')
