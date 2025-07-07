import argparse
import json
import sys
import time
import os
import logging
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from torch import nn
from torch.utils.data import DataLoader
from util.utils import mkdir_if_needed
from util.read_ecg import get_ecg_transforms

# if neptune exists, import it
try:
    import neptune
except ImportError:
    print("Neptune is not installed. Please install it to use this feature by running 'pip install neptune'.")
    neptune = None

from model.load_model import load_ecg_model
from trainer_ecg import trainer
from util.load_data import load_ukb_ecg_survival_data
from util.datasets import UKBSurvivalDataset
from util.utils import init_seeds, save_args


parser = argparse.ArgumentParser()

# Dataset parameters
parser.add_argument("--image_path", default="/data1/yifan/ecg/data/ukb/ecg_npy",
                    help="dataset directory for images, npy files or xml files")
parser.add_argument("--data_type", default="npy", choices=["npy", "xml"],
                    help="data type: npy or xml")
parser.add_argument("--dataset", default="ukb")
parser.add_argument('--k_fold', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=32)
parser.add_argument('--group', type=str, default='insomnia_score')
parser.add_argument('--time', type=str, default='time', help='time to event')
parser.add_argument('--event', type=str, default='event', help='name of event')
parser.add_argument('--shuffle', type=int, default=0, help='whether to shuffle the data')

# model config path
parser.add_argument('--config', type=str, default='config/st_mem.yaml', help='/path/to/config/file.yaml')
parser.add_argument("--freeze", type=int, default=0, help="whether to freeze the backbone model")

# Training parameters
parser.add_argument("--batch_size", type=int, default=320, help="input batch size for training")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--warmup_lr", type=float, default=1e-7, help="learning rate for warmup")
parser.add_argument("--warmup_epochs", type=int, default=2, help="number of warmup epochs")
parser.add_argument("--lr_scheduler", type=str, default="warmup_cosine",
                    choices=["warmup_cosine", "cosineannealing", "step"],
                    help="learning rate scheduler")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
parser.add_argument("--max_epochs", type=int, default=100, help="number of epochs to train")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--device", type=str, default="cuda", help="device for computation (default: cuda)")
parser.add_argument("--save_every", type=int, default=1, help="save every n epochs")
parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus to use for training")

# loss parameters
parser.add_argument("--w1", type=float, default=1.0, help="weight for the contrastive loss, 0.0 to disable")
parser.add_argument("--w2", type=float, default=1.0, help="weight for the proportion loss, 0.0 to disable")
parser.add_argument("--w3", type=float, default=0.5, help="weight for the entropy loss, 0.0 to disable")
parser.add_argument("--w4", type=float, default=0.5, help="weight for the balance loss, 0.0 to disable")
parser.add_argument("--minp", type=float, default=0.35, help="minimum proportion of patients in first cluster, in [0, 1)")
parser.add_argument("--temperature", type=float, default=0.3, help="temperature for output logits")
# use neptune for logging
parser.add_argument("--neptune", type=int, default=0, help="whether to use neptune for logging")
parser.add_argument("--neptune_config", type=str, default="config/neptune.json",
                    help="neptune config file path, if neptune is used")
# save path for checkpoints and logs
parser.add_argument("--output_dir", type=str, default="checkpoints", help="output directory")
parser.add_argument("--log_every_n_epoch", type=int, default=1, help="log every n epochs")

args = parser.parse_args()


def main():
    seed = args.seed
    init_seeds(seed)
    device = torch.device(args.device)

    image_path = args.image_path
    dataset_name = args.dataset
    group_name = args.group
    time_name = args.time
    event_name = args.event

    df_data = pd.read_csv('data/ukb_ecg_data_balanced.csv')

    ids = df_data['Eid'].values
    groups = df_data[group_name].values
    events = df_data[event_name].values

    print(df_data[group_name].value_counts())

    # setup neptune logging
    if args.neptune and neptune is not None:
        if not os.path.exists(args.neptune_config):
            print(f"Neptune config file {args.neptune_config} does not exist. Please provide a valid path.")
        else:
            with open(args.neptune_config, 'r') as f:
                neptune_config = json.load(f)
            print('loaded neptune config successfully')
            try:
                logger = neptune.init_run(
                    project=neptune_config['project'],
                    api_token=neptune_config['api_token'],
                    tags=[dataset_name, group_name, str(args.minp), str(args.seed)],
                    name=f'STMEM_{dataset_name}_{group_name}',
                )
            except:
                print("Failed to initialize Neptune. Please check your API token and project name.")
                logger = None
    else:
        logger = None

    save_path = os.path.join(args.output_dir, f'ecg_{dataset_name}_{group_name}_lr{args.lr}_minp{args.minp}_seed{seed}')
    mkdir_if_needed(save_path)

    # save args
    save_args(args, save_path=save_path, save_name=f'args_{dataset_name}_{group_name}.txt')

    # local logging
    log_name = f'log_{time.strftime("%Y%m%d_%H%M%S")}.txt'
    logging.basicConfig(filename=os.path.join(save_path, log_name),
                        level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    kf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=seed)
    fold = 1
    train_transforms, val_transforms = get_ecg_transforms()

    for train_idx, test_idx in kf.split(ids, groups):

        train_ids, test_ids = ids[train_idx], ids[test_idx]
        # pass fola 1, only train fold 2
        # if fold == 1:
        #     fold += 1
        #     continue

        df_train = df_data.loc[df_data['Eid'].isin(train_ids)]
        df_test = df_data.loc[df_data['Eid'].isin(test_ids)]
        train_data = load_ukb_ecg_survival_data(df_train, group=group_name, time=time_name, event=event_name,
                                                clinical=True,
                                                random_state=seed, shuffle=args.shuffle)
        test_data = load_ukb_ecg_survival_data(df_test, group=group_name, time=time_name, event=event_name,
                                               clinical=True,
                                               random_state=seed, shuffle=args.shuffle)

        print(f'Number of training data: {len(df_train)}, test data: {len(df_test)}')

        train_dataset = UKBSurvivalDataset(image_path, train_data, transform=train_transforms, data_type=args.data_type)
        test_dataset = UKBSurvivalDataset(image_path, test_data, transform=val_transforms, data_type=args.data_type)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = load_ecg_model(args.config, args.freeze)
        if args.n_gpus > 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if args.n_gpus > torch.cuda.device_count():
                print(f"Warning: {args.n_gpus} GPUs requested, but only {torch.cuda.device_count()} available. "
                      f"Using {torch.cuda.device_count()} GPUs.")
                gpu_used = torch.cuda.device_count()
            else:
                gpu_used = args.n_gpus
            model = nn.DataParallel(model)
            print(f"Using {gpu_used} GPUs for training.")
        model = model.to(device)
        print(f"Model loaded to {device}.")

        snapshot_path = os.path.join(save_path, f'fold{fold}')
        output_fig_path = os.path.join(save_path, f'fold{fold}', 'figs')
        cluster_path = os.path.join(save_path, f'fold{fold}', 'clusters')
        mkdir_if_needed(snapshot_path)
        mkdir_if_needed(output_fig_path)
        mkdir_if_needed(cluster_path)

        logging.info('-----------------------------Training fold: {}--------------------------------'.format(fold))
        trainer(args, logger, model, train_loader, test_loader, device, fold, snapshot_path)

        fold += 1

if __name__ == '__main__':
    main()