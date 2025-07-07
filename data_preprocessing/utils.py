import numpy as np
import pandas as pd

from statsmodels.formula.api import logit
from scipy.optimize import linear_sum_assignment



def calc_smd(df, variable, type):
    treated = df[df[type] == 1]
    control = df[df[type] == 0]
    mean_treated = treated[variable].mean()
    mean_control = control[variable].mean()
    std = df[variable].std()
    smd = np.abs(mean_treated - mean_control) / std
    return smd


def psm(df, type, covars, need_shuffle=False, seed=42, rate=1):
    """
    Propensity Score Matching (PSM) using logistic regression and Hungarian algorithm for optimal matching.
    :param df: dataframe containing the data
    :param type: depending variable (treatment variable) e.g., something like disease status, e.g. 't1dm'
    :param covars: covariates to be used for matching, i.e. ['age', 'sex', 'bmi'], at least 1 covariate is needed
    :param need_shuffle: whether to shuffle the matched dataframe
    :param seed: random seed for shuffling
    :param rate: the ratio of treated to control samples, default is 1:1
    :return: balanced dataframe and matched dataframe
    """
    df = df.copy()
    # dropna
    df = df.dropna(subset=covars + [type])
    print(len(df), 'rows after dropping na....')
    if len(covars) == 0:
        raise ValueError('At least one covariate is needed for matching')
    # calculate propensity score
    covars_formula = ' + '.join(covars)
    model = logit(f"{type} ~ {covars_formula}", df).fit()
    df['propensity_score'] = model.predict(df)

    # match samples
    df_1 = df[df[type] == 1].copy()
    df_0 = df[df[type] == 0].copy()
    print(len(df_0), 'control samples, ', len(df_1), 'treated samples')

    score_matrix = abs(df_1['propensity_score'].values[:,None] - df_0['propensity_score'].values)
    row_ind,col_ind = linear_sum_assignment(score_matrix)

    df_balanced = pd.concat([df_1.iloc[row_ind], df_0.iloc[col_ind]], axis=0)

    if need_shuffle:
        # shuffle the balanced dataframe to avoid any order bias
        df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df, df_balanced
