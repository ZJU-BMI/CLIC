import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.load_model import load_ecg_model
from util.datasets import UKBSurvivalDataset
from util.load_data import load_ukb_ecg_survival_data
from util.read_ecg import get_ecg_transforms
from util.utils import load_args, init_seeds, mkdir_if_needed
from sklearn.model_selection import StratifiedKFold


args = load_args('args_ukb_insomnia_score.txt')
seed = args.seed
init_seeds(seed)
image_path = args.image_path
dataset_name = args.dataset
group_name = args.group
time_name = args.time
event_name = args.event
device = torch.device(args.device)
output_dir = args.output_dir

df_data = pd.read_csv('data/ukb_ecg_data_balanced.csv')
# drop na
df_data = df_data[~(df_data[group_name].isna())]

ids = df_data['Eid'].values
groups = df_data[group_name].values

kf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
fold = 1

# checkpoint save path
save_path = os.path.join(output_dir, f'ECG_{dataset_name}_{group_name}_lr{args.lr}_minp{args.minp}')

train_transforms, val_transforms = get_ecg_transforms()


for train_idx, test_idx in kf.split(ids, groups):
    train_ids, test_ids = ids[train_idx], ids[test_idx]

    df_train = df_data.loc[df_data['Eid'].isin(train_ids)]
    df_test = df_data.loc[df_data['Eid'].isin(test_ids)]

    train_data = load_ukb_ecg_survival_data(df_train, group=group_name, time=time_name, event=event_name,
                                            random_state=seed, shuffle=args.shuffle)
    test_data = load_ukb_ecg_survival_data(df_test, group=group_name, time=time_name, event=event_name,
                                           random_state=seed, shuffle=args.shuffle)

    print(f'Number of training data: {len(df_train)}, test data: {len(df_test)}')

    train_dataset = UKBSurvivalDataset(image_path, train_data, transform=train_transforms, data_type=args.data_type)
    test_dataset = UKBSurvivalDataset(image_path, test_data, transform=val_transforms, data_type=args.data_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_ecg_model(args.config)
    # load trained model
    checkpoint = os.path.join(save_path, f'best_fold{fold}.pth')
    if os.path.exists(checkpoint):
        train_dict = torch.load(checkpoint, map_location='cpu')
        msg = model.load_state_dict(train_dict['state_dict'])
        print(f'Loaded model from {checkpoint}: {msg}')
    else:
        print(f'Checkpoint not found: {checkpoint}')
        continue

    model = model.to(device)
    model.eval()

    with torch.no_grad():

        cluster_assigns = []
        labels = []
        groups = []
        sample_ids = []
        predictions = []
        times = []
        events = []

        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=125):
            image = batch['image'].to(device)
            group = batch['group']
            time = batch['time'].to(device)
            event = batch['event'].to(device)
            lambda_ = batch['lambda'].to(device)
            sample_id = batch['sample_id']

            if len(image.shape) == 4:
                # dim of test image is [b, n, 12, 2250], n is the number of crops
                # average the output over n crops
                output = []
                for i in range(image.shape[1]):
                    output.append(model(image[:, i]))
                output = torch.stack(output, dim=1)
                output = output.mean(dim=1)  # average over n slices
            else:
                output = model(image)

            pred_probs = torch.softmax(output / args.temperature, dim=1).detach().cpu().numpy()
            cluster_assign = output.argmax(dim=1).detach().cpu().numpy()

            cluster_assigns.append(cluster_assign)
            groups.append(group.cpu().numpy())
            sample_ids.append(sample_id.cpu().numpy())
            predictions.append(pred_probs)
            times.append(time.cpu().numpy())
            events.append(event.cpu().numpy())

        cluster_assigns = np.concatenate(cluster_assigns)
        groups = np.concatenate(groups)
        sample_ids = np.concatenate(sample_ids)
        predictions = np.concatenate(predictions)
        times = np.concatenate(times)
        events = np.concatenate(events)

        # save the results
        df = pd.DataFrame({
            'Eid': sample_ids,
            group_name: groups,
            'cluster_assign': cluster_assigns,
            'pred_cluster0': predictions[:, 0],
            'pred_cluster1': predictions[:, 1],
            'time': times,
            'event': events,
            'fold': fold
        })

        if fold == 1:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], axis=0)
    fold += 1

res_path_root = 'results/non_insomnia'
mkdir_if_needed('results')
df_all.to_csv(res_path_root + '/ukb_cluster_assignments.csv', index=False)
df = pd.read_csv('data/ukb_ecg_data_balanced.csv')
df = pd.merge(df_all[['Eid', 'cluster_assign', 'fold']], df, on='Eid', how='left')
df.to_csv(os.path.join(res_path_root, 'ukb_ecg_data_balanced_cluster.csv'), index=False)