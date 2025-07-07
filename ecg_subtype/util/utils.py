import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from monai import transforms
from sklearn.manifold import TSNE

from torch import Tensor, nn
from typing import Optional
import torch.nn.functional as F

from torch.backends import cudnn
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import neurokit2 as nk


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def normalize_data(x, norm_mode='standard'):
    if isinstance(x, np.ndarray):
        num_Patient, num_Feature = np.shape(x)
        if norm_mode is None:
            return x
        if norm_mode == 'standard':  # zero mean unit variance
            for j in range(num_Feature):
                if np.std(x[:, j]) != 0:
                    x[:, j] = (x[:, j] - np.mean(x[:, j])) / np.std(x[:, j])
                else:
                    x[:, j] = (x[:, j] - np.mean(x[:, j]))
        elif norm_mode == 'minmax':  # min-max normalization
            for j in range(num_Feature):
                x[:, j] = (x[:, j] - np.min(x[:, j])) / (np.max(x[:, j]) - np.min(x[:, j]))
        else:
            print("INPUT MODE ERROR!")
    elif isinstance(x, pd.DataFrame):
        for col in x.columns:
            if norm_mode == 'standard':
                x[col] = (x[col] - np.mean(x[col])) / (np.std(x[col]) + 1e-9)  # add small value to avoid division by zero
            elif norm_mode == 'minmax':
                x[col] = (x[col] - np.min(x[col])) / (np.max(x[col]) - np.min(x[col]))
            else:
                print("INPUT MODE ERROR!")
    return x


def mkdir_if_needed(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def to_device(device, *arrays):
    if len(arrays) == 0:
        return None
    result = [array.to(device) for array in arrays]
    return tuple(result)


def save_args(args, save_path, save_name=None):
    if save_name is None:
        save_name = 'args.txt'
    with open(os.path.join(save_path, save_name), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)


def load_args(args_path):
    with open(args_path, 'r') as fp:
        args = json.load(fp)
    args = argparse.Namespace(**args)
    return args


def lambdas_(time, event):
    """
    lambda_i= Sum_t Omega_t / N_t * I(T_i>t)
    y is survival time with shape (n, )
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=time, event_observed=event)  # fit km to your data

    # extract event table
    e_table = pd.DataFrame(kmf.event_table).reset_index()  # reset index so that time is part of the data.
    omegat_over_Nt = e_table["observed"] / e_table["at_risk"]  # Omege_t/N_t term in the equation.

    lamdas = np.zeros(time.shape[0])  # where to save lambda values
    for ix, Ti in enumerate(time):
        lamdas[ix] = np.sum(omegat_over_Nt * (Ti > e_table["event_at"]))
    return lamdas


def show_pvalue(pval, show_sig_marker=False, marker1='*', marker2='**', marker3='***', marker_ns=' (NS)'):
    if pval >= 0.1:  # [0.1, 1)
        res = f'{pval:.2f}'
    elif pval >= 0.01: # [0.01, 0.1)
        res = f'{pval:.3f}'
    elif pval >= 0.001: # [0.001, 0.01)
        res = f'{pval:.4f}'
    else: # [0, 0.001)
        res = f'{pval:.2e}'
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
                  fold=0, epoch=0, label1='Group 1', label2='Group 2', title=None, save_path=None):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(6, 5))

    # plot the survival function for the first group
    if len(time) > 0:
        N1 = len(time)
        kmf.fit(time, event, label='Group 1')
        kmf.plot_survival_function(ax=ax, label=f'{label1} (N={N1:,})', color='red')

    # plot the survival function for the second group if provided
    if time2 is not None and event2 is not None:
        if len(time2) > 0:
            N2 = len(time2)
            kmf2 = KaplanMeierFitter()
            kmf2.fit(time2, event2, label='Group 2')
            kmf2.plot_survival_function(ax=ax, label=f'{label2} (N={N2:,})', color='lightblue')
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

    return fig, ax


if __name__ == '__main__':
    # Example usage
    df = pd.read_csv('../data/ukb_ecg_data_balanced.csv')
    ecgs = df['ECG'].values