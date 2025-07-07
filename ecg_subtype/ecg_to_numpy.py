from os import makedirs
from sys import exception

import numpy as np
import pandas as pd
from util.read_ecg import get_ecg, get_ecg_transforms
import os

#
ecg_path_root = '/mnt/data/ukb_heartmri/ukb_20205/'
ecgs = os.listdir(ecg_path_root)
train_transforms, _ = get_ecg_transforms()
print(len(ecgs))

# check all ecg files if they are valid
valid_eids = []
invalid_eids = []
makedirs('/data1/yifan/ecg/data/ukb/ecg_npy', exist_ok=True)

for i, ecg_filename in enumerate(ecgs):
    if ecg_filename.endswith('_2_0.xml'): # Eid_20205_2_0.xml
        ecg_path = os.path.join(ecg_path_root, ecg_filename)
        try:
            ecg_data = get_ecg(ecg_path)
            # ecg_data_preprocessed = train_transforms(ecg_data)
        except exception as e:
            print(f"Error processing {ecg_filename}, invalid ECG data, error: {e}")
            invalid_eids.append(ecg_filename.split('_')[0])
            continue

        eid = ecg_filename.split('_')[0]
        # save
        if not os.path.exists(f'/data1/yifan/ecg/data/ukb/ecg_npy/{eid}_20205_2_0.npy'):
            np.save(f'/data1/yifan/ecg/data/ukb/ecg_npy/{eid}_20205_2_0.npy', ecg_data)
            print(f'{i}/{len(ecgs)}: {ecg_filename} saved to {eid}_20205_2_0.npy')
        else:
            print(f'{i}/{len(ecgs)}: {eid}_20205_2_0.npy already exists')

        valid_eids.append(int(eid))



# save valid_eids
df_valid = pd.DataFrame({
    'eid': valid_eids,
    'ECG': [f'{eid}_20205_2_0.npy' for eid in valid_eids]
})
df_valid.to_csv('/data1/yifan/ecg/data/ukb/ecg_valid.csv', index=False)
print('number of valid eids:', len(valid_eids))
print('number of invalid eids:', len(invalid_eids))