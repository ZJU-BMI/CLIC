import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from torch.utils.data import Dataset, DataLoader
from util.read_ecg import get_ecg, get_ecg_transforms


class UKBSurvivalDataset(Dataset):
    def __init__(self, image_path: str, survival_data: Dict, transform=None, data_type='npy'):
        self.image_path = image_path
        self.transform = transform

        self.ecg_data = survival_data['ecg']
        self.group = survival_data.get('group', None)
        self.event = survival_data.get('event', None)
        self.time = survival_data.get('time', None)
        self.sample_ids = survival_data.get('sample_id', None)
        self.lambdas = survival_data.get('lambda', None)
        self.clinical_data = survival_data.get('clinical', None)
        self.data_type = data_type


        # make sure event and time are provided together

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):

        ecg_path = os.path.join(self.image_path, self.ecg_data[idx])
        if self.data_type == 'npy':
            ecg = np.load(ecg_path, allow_pickle=True) # load as npy file
        else:
            ecg = get_ecg(ecg_path) # load as xml file

        if self.transform:
            ecg = self.transform(ecg)

        sample = {
            'image': ecg,
        }

        if self.sample_ids is not None:
            sample_id = self.sample_ids[idx]
            sample['sample_id'] = sample_id

        if self.event is not None:
            event = self.event[idx]
            sample['event'] = event

        if self.time is not None:
            time = self.time[idx]
            sample['time'] = time

        if self.lambdas is not None:
            lambdas = self.lambdas[idx]
            sample['lambda'] = lambdas

        if self.group is not None:
            group = self.group[idx]
            sample['group'] = group

        if self.clinical_data is not None:
            clinical_data = self.clinical_data[idx]
            sample['clinical'] = clinical_data

        return sample


if __name__ == '__main__':
    image_path = '/data1/yifan/ecg/data/ukb/ecg_npy'
    df = pd.read_csv('../data/ukb_ecg_data_balanced.csv')
    ecg_data = df['ECG'].values
    times = df['time'].values
    event = df['event'].values
    group = df['insomnia_score'].values
    clinical_cols = ['Sex']
    clinical_data = df[clinical_cols].values
    train_transforms, _ = get_ecg_transforms()
    survival_data = {
        'ecg': ecg_data,
        'time': times,
        'event': event,
        'group': group,
        'clinical': clinical_data
    }
    dataset = UKBSurvivalDataset(image_path, survival_data, transform=train_transforms, data_type='npy')

    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    for i, sample in enumerate(loader):
        print(sample['image'].shape)
        print(sample['time'].shape)
        print(sample['group'].shape)
        print(sample['event'].shape)
        # print(sample['lambda'].shape)
        # if 'clinical_data' in sample:
        print(sample['clinical'].shape)

        if i == 1:
            break



