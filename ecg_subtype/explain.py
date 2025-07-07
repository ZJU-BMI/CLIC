import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.load_model import load_ecg_model
from model.losses import ContrastiveLoss
from util.datasets import UKBSurvivalDataset
from util.load_data import load_ukb_ecg_survival_data
from util.read_ecg import get_ecg_transforms
from util.utils import load_args, init_seeds, mkdir_if_needed
from sklearn.model_selection import StratifiedKFold


def compute_saliency_maps(args, model, test_loader):
    device = args.device
    saliency_maps = []
    sample_ids = []
    contrastive_loss = ContrastiveLoss(w1=1, w2=0., minp=0., temperature=args.temperature)
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), ncols=125):
        image = batch['image'].to(device)
        group = batch['group'].to(device)
        time = batch['time'].to(device)
        event = batch['event'].to(device)
        lambda_ = batch['lambda'].to(device)
        sample_id = batch['sample_id']

        sample_ids.append(sample_id.cpu().numpy())
        # Set requires_grad=True to compute gradients
        image.requires_grad_()
        # Forward pass
        if len(image.shape) == 4:
            saliency = []
            for i in range(image.shape[1]):
                output = model(image[:, i])
                loss = contrastive_loss(output, time, event, lambda_, group, softmax=True)
                model.zero_grad()
                loss.backward()
                saliency_ = image.grad.data[:, i].abs()
                saliency.append(saliency_)
            saliency = torch.stack(saliency, dim=1)

        else:
            output = model(image)
            loss = contrastive_loss(output, time, event, lambda_, group, softmax=True)
            model.zero_grad()
            loss.backward()
            saliency = image.grad.data.abs()

        saliency_maps.append(saliency.detach().cpu().numpy())

    saliency_maps = np.concatenate(saliency_maps, axis=0) # [n, 3, 12, 2250]
    sample_ids = np.concatenate(sample_ids, axis=0)
    return saliency_maps, sample_ids


def merge_saliency_maps_crops(kfold, norm=True):
    # saliency_maps: [n, 3, 12, 2250]
    num_segments, num_leads, crop_length = 3, 12, 2250
    start_idx = np.arange(start=0,
                          stop=5000 - crop_length + 1,
                          step=(5000 - crop_length) // (num_segments - 1))
    print(start_idx)
    # map cropped ecg to the original ecg
    saliency_maps = []
    for fold in range(1, kfold + 1):
        saliency = np.load(f'results/insomnia_score/saliency_maps/saliency_map_fold{fold}.npy')
        x = np.zeros((len(saliency), 12, 5000), dtype=np.float32)
        for i, idx in enumerate(start_idx):
            x[:, :, idx:idx + crop_length] = np.maximum(x[:, :, idx:idx + crop_length], saliency[:, i])
        # min-max normalize the saliency to [0, 1]
        if norm:
            for i in range(num_leads):
                # min-max on each lead
                # x [n, 12, 2250],
                min_x = np.min(x[:, i], axis=1, keepdims=True)
                max_x = np.max(x[:, i], axis=1, keepdims=True)
                x[:, i] = (x[:, i] - min_x) / (max_x - min_x + 1e-9)

        saliency_maps.append(x)
        # free up memory
        del saliency

    # save saliency maps for original ecg
    saliency_maps = np.concatenate(saliency_maps, axis=0)
    if norm:
        np.save('results/insomnia_score/saliency_maps/saliency_map_original_normalized.npy', saliency_maps)
        np.save('results/insomnia_score/saliency_maps/saliency_map_original_example_normalized.npy', saliency_maps[0:100])
    else:
        np.save('results/insomnia_score/saliency_maps/saliency_map_original.npy', saliency_maps)
        np.save('results/insomnia_score/saliency_maps/saliency_map_original_example.npy', saliency_maps[0:100])


def main(args):
    seed = args.seed
    init_seeds(seed)
    image_path = args.image_path
    dataset_name = args.dataset
    group_name = args.group
    time_name = args.time
    event_name = args.event
    device = torch.device(args.device)

    df_data = pd.read_csv('data/ukb_ecg_data_balanced.csv')
    # drop na
    df_data = df_data[~(df_data[group_name].isna())]

    ids = df_data['Eid'].values
    groups = df_data[group_name].values

    kf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    fold = 1
    # checkpoint save path
    save_path = os.path.join(args.output_dir, f'{dataset_name}_{group_name}_lr{args.lr}_minp{args.minp}_seed{seed}')

    train_transforms, val_transforms = get_ecg_transforms()

    for train_idx, test_idx in kf.split(ids, groups):
        train_ids, test_ids = ids[train_idx], ids[test_idx]
        df_train = df_data.loc[df_data['Eid'].isin(train_ids)]
        df_test = df_data.loc[df_data['Eid'].isin(test_ids)]

        train_data = load_ukb_ecg_survival_data(df_train, group=group_name, time=time_name, event=event_name,
                                                random_state=seed, shuffle=args.shuffle)
        test_data = load_ukb_ecg_survival_data(df_test, group=group_name, time=time_name, event=event_name,
                                               random_state=seed, shuffle=args.shuffle)

        print(f'Number of training ECGs: {len(train_data['ecg'])}, test ECGs: {len(test_data['ecg'])}')

        train_dataset = UKBSurvivalDataset(image_path, train_data, transform=val_transforms, data_type=args.data_type)
        test_dataset = UKBSurvivalDataset(image_path, test_data, transform=val_transforms, data_type=args.data_type)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = load_ecg_model(args.config)

        # load trained model
        checkpoint = os.path.join(save_path, f'best_fold{fold}.pth')
        if os.path.exists(checkpoint):
            train_dict = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(train_dict['state_dict'])
            print(f'Loaded model from {checkpoint}')
        else:
            print(f'Checkpoint not found: {checkpoint}')
            continue

        model = model.to(args.device)
        model = nn.DataParallel(model, device_ids=[0, 1]) # use all GPUs

        model.eval()

        # compute saliency maps
        saliency_maps, sample_ids = compute_saliency_maps(args, model, test_loader)

        # save saliency maps
        np.save(f'results/insomnia_score/saliency_maps/saliency_map_fold{fold}.npy', saliency_maps)
        df_samples = pd.DataFrame({
            'Eid': sample_ids,
            'Fold': fold,
        })

        if fold == 1:
            df_all_samples = df_samples
        else:
            df_all_samples = pd.concat([df_all_samples, df_samples], axis=0)
        fold += 1

    print('saliency map compute completed, merging results...')

    # save all samples
    df_all_samples.to_csv('results/insomnia_score/saliency_maps/samples_ids.csv', index=False)


if __name__ == '__main__':
    args = load_args('args_ukb_insomnia_score.txt')
    mkdir_if_needed('results/insomnia_score/saliency_maps')
    main(args)
    merge_saliency_maps_crops(kfold=args.k_fold, norm=False)

