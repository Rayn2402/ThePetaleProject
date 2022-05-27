"""
Filename: warmup_figures.py

Author: Nicolas Raymond

Description: This file is used to create figures that compare VO2 peak model predictions on the holdout set

Date of last modification: 2022/05/04
"""

import matplotlib.pyplot as plt

from os.path import join
from pandas import read_csv
from settings.paths import Paths
from src.data.extraction.constants import PARTICIPANT, SEX
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.sampling import get_warmup_data

if __name__ == '__main__':

    # We read the prediction data
    df = read_csv(join(Paths.CSV_FILES, "warmup_predictions.csv"), index_col=0)

    # We load the data that contains the sex of the participants
    sex_df, _, _, _, = get_warmup_data(PetaleDataManager(), sex=True, holdout=True)
    sex_df.set_index(PARTICIPANT, inplace=True)
    sex_df = sex_df[SEX]

    # We merge the sex data to the predictions df
    df = df.join(sex_df)

    # Enable LaTeX
    plt.rc('text', usetex=True)

    # We create the subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)

    # We set the axes
    men_df, women_df = df.loc[df[SEX] == 'Men'], df.loc[df[SEX] == 'Women']

    axes[0].plot(df.index, df['target'], 'x', color='grey', label='Targets')
    axes[0].plot(men_df.index, men_df['OE'], 'd', color='#1f77b4', label='Last equation (Men)')
    axes[0].plot(women_df.index, women_df['OE'], 'o', color='#1f77b4',  label='Last equation (Women)')
    axes[0].legend(loc='lower right')

    axes[1].plot(df.index, df['target'], 'x', color='grey', label='Targets')
    axes[1].plot(men_df.index, men_df['GAT'], 'd', color='orange', label='GAT (Men)')
    axes[1].plot(women_df.index, women_df['GAT'], 'o', color='orange', label='GAT (Women)')
    axes[1].legend(loc='lower right')

    for ax in axes:
        ax.xaxis.set_tick_params(rotation=90)

    # We set the titles
    fig.supylabel('VO$_{2}$ peak (ml/kg/min)')
    fig.supxlabel('Survivors in the holdout set', x=0.528)

    # We set the labels
    # handles, labels = [(a + b) for a, b in zip(axes[0].get_legend_handles_labels(), axes[1].get_legend_handles_labels())]
    # fig.legend(handles, labels, bbox_to_anchor=(0.9875, 0.40))

    # Figure saving
    fig.tight_layout()
    for f in ['pdf', 'svg']:
        plt.savefig(f'warmup_predictions.{f}')
