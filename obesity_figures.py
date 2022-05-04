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

    # We create the subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)

    # We set the axes
    axes[0].plot(df.index, df['target'], 'o', color='grey', label='Targets')
    axes[0].plot(df.index, df['Enet'], 'o', label='Linear Reg. (B)')
    axes[0].plot(df.index, df['GGAE'], 'o', label='Linear Reg. + GGAE (B + G)')

    axes[1].plot(df.index, df['target'], 'o', color='grey')
    axes[1].plot(df.index, df['XGBoost'], 'o', label='XGBoost (B + G)', color='green')

    for ax in axes:
        ax.xaxis.set_tick_params(rotation=90)

    # We set the titles
    fig.supylabel('Total Body Fat (\\%)')
    fig.supxlabel('Participants', x=0.528)

    # We set the labels
    handles, labels = [(a + b) for a, b in zip(axes[0].get_legend_handles_labels(), axes[1].get_legend_handles_labels())]
    fig.legend(handles, labels, bbox_to_anchor=(0.31, 0.96))

    # Figure saving
    fig.tight_layout()
    for f in ['pdf', 'svg']:
        plt.savefig(f'obesity_predictions.{f}')
