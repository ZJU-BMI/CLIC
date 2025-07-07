import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def mkdir_if_needed(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def to_scientific_notation(x, precision=2):
    """
    Convert a number to scientific notation with a specified precision.
    """
    if x == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return r'$' + f'{mantissa:.{precision}f}' + r'\times 10^{' + f'{exponent}' + r'}$'


def show_pvalue(pval, show_sig_marker=False, marker1='*', marker2='**', marker3='***', marker_ns=' (NS)', scientific=False):
    if pval == 0:
        return '< 2.23e-308'  # This is the smallest positive float in Python
    if pval >= 0.1:  # [0.1, 1)
        res = f'{pval:.2f}'
    elif pval >= 0.01: # [0.01, 0.1)
        res = f'{pval:.3f}'
    elif pval >= 0.001: # [0.001, 0.01)
        res = f'{pval:.4f}'
    else: # [0, 0.001)
        res = f'{pval:.2e}'
        if scientific:
            res = to_scientific_notation(pval, precision=2)
    if show_sig_marker:
        if pval < 0.001:
            sig_marker = marker3
        elif pval < 0.01:
            sig_marker = marker2
        elif pval < 0.05:
            sig_marker = marker1
        else:
            sig_marker = marker_ns
        res = f'{res}{sig_marker}'

    return res


def survival_plot(logger, time, event, time2=None, event2=None, group='subject', stage='train',
                  color1='red', color2='lightblue',
                  fold=0, epoch=0, label1='Group 1',
                  label2='Group 2', title=None, save_path=None):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(6, 5))

    # plot the survival function for the first group
    if len(time) > 0:
        N1 = len(time)
        kmf.fit(time, event, label='Group 1')
        kmf.plot_survival_function(ax=ax, label=f'{label1} (N={N1:,})', color=color1)

    # plot the survival function for the second group if provided
    if time2 is not None and event2 is not None:
        if len(time2) > 0:
            N2 = len(time2)
            kmf2 = KaplanMeierFitter()
            kmf2.fit(time2, event2, label='Group 2')
            kmf2.plot_survival_function(ax=ax, label=f'{label2} (N={N2:,})', color=color2)
            log_rank = logrank_test(time, time2, event_observed_A=event, event_observed_B=event2)

        # add log rank p value to the plot in the bottom left corner
        if len(time) > 0 and len(time2) > 0:
            ax.text(0.02, 0.06, f'Log Rank p-value: {show_pvalue(log_rank.p_value)}',
                    fontsize=12, transform=ax.transAxes)

    plt.title(f'Survival Plot for {group} Epoch {epoch}' if title is None else title, fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Survival Probability', fontsize=14)
    ax.tick_params(labelsize=14, axis='both', which='major')
    plt.grid(False)

    if logger is not None:
        logger[f"{stage}/fold{fold}/Survival_Plot_{group}"].append(fig)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"Survival_Plot_{stage}_{group}_{epoch}.jpg"), bbox_inches='tight', dpi=300)

    if logger is None:
        return fig, ax

    plt.close(fig)

