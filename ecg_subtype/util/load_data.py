import os
from typing import Optional
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from util.utils import normalize_data
from sklearn.model_selection import KFold
from util.utils import lambdas_
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def load_ukb_dataset(df: pd.DataFrame,
                     group: str = 'insomnia_score', # group by group name
                     label_name: Optional[str] = 'PHQ_9', # label name in the csv file
                     random_state: Optional[int] = 42,
                     shuffle: bool = False):
    if shuffle:
        # shuffle the data
        df = df.sample(frac=1, random_state=random_state)
    sample_ids = df['Eid'].values
    ecg_data = df['ECG'].values
    groups = df[group].values

    if label_name is not None and label_name in df.columns:
        label = df[label_name].values

    return_dict = {
        'sample_ids': sample_ids,
        'groups': groups,
        'ecg': ecg_data,
        'label': label
    }
    return return_dict


def load_ukb_ecg_survival_data(
        df: pd.DataFrame,
        group: str = 'insomnia_score', # group by group name
        event: str = 'event', # event name in the csv file
        time: str = 'time', # time name in the csv file
        clinical: Optional[bool] = None, # clinical data name in the csv file
        random_state: Optional[int] = 42,
        shuffle: bool = False,
):
    '''
    Load the UKB ECG with survival data from the csv file.
    '''
    if shuffle:
        # shuffle the data
        df = df.sample(frac=1, random_state=random_state)
    sample_ids = df['Eid'].values
    ecg_data = df['ECG'].values
    groups = df[group].values

    event = df[event].values
    ev_time = df[time].values

    lambdas = lambdas_(ev_time, event)

    if clinical is not None:
        clinical_cols = ['Sex']
        # # one-hot encode for categorical variables
        df_clinical_data = df[clinical_cols]
        clinical_data = df_clinical_data.values.astype(np.float32)
    else:
        clinical_data = None

    return_dict = {
        'sample_id': sample_ids,
        'group': groups,
        'ecg': ecg_data,
        'event': event,
        'time': ev_time,
        'lambda': lambdas,
        'clinical': clinical_data
    }
    return return_dict


if __name__ == '__main__':
    # Example usage
    df_data = pd.read_csv('../data/ukb_ecg_data_balanced.csv')
    data = load_ukb_ecg_survival_data(df_data, group='insomnia_score', event='event', time='time', clinical=True, random_state=42)

    print(data['sample_id'][:5])
    print(data['group'][:5])
    print(data['ecg'][:5])
    print(data['event'][:5])
    print(data['time'][:5])
    print(data['lambda'][:5])
    if data['clinical'] is not None:
        print(data['clinical'][:5])
