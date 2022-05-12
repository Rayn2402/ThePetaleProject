"""
Filename: obesity_figures.py

Author: Nicolas Raymond

Description: This file is used to create figures that compare obesity model predictions on the holdout set

Date of last modification: 2022/05/04
"""

import matplotlib.pyplot as plt

from os.path import join
from pandas import read_csv
from settings.paths import Paths

if __name__ == '__main__':

    # We read the data
    df = read_csv(join(Paths.CSV_FILES, "obesity_predictions.csv"), index_col=0)

    # Enable LaTeX
    plt.rc('text', usetex=True)

    # Figure creation
    plt.plot(df.index, df['target'], 'o', color='grey', label='Targets')
    plt.plot(df.index, df['Enet'], 'o', label='Linear Reg. (B)')
    plt.plot(df.index, df['GGAE'], 'o', label='Linear Reg. + GGAE (B + G)')
    plt.xticks(rotation=90)

    plt.ylabel('Total Body Fat (\\%)')
    plt.xlabel('Participants')
    plt.legend()
    plt.tight_layout()

    for f in ['pdf', 'svg']:
        plt.savefig(f'obesity_predictions.{f}')
