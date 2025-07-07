import os
from argparse import Namespace
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.losses import ContrastiveLoss, EntropyLoss
from util.utils import survival_plot
from util.lr_scheduler import LinearWarmupCosineAnnealingLR


def trainer(args: Namespace, logger, model, train_loader, test_loader,
            device: torch.device, fold=1, snapshot_path=None):

    lr = args.lr
    num_epochs = args.max_epochs

    contrastive_loss = ContrastiveLoss(w1=args.w1, w2=args.w2, minp=args.minp, temperature=args.temperature)
    entropy_loss = EntropyLoss(weight=args.w3, temperature=args.temperature)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=num_epochs,
                                                  warmup_start_lr=args.warmup_lr)
    else:
        # not implemented other schedulers yet
        scheduler = None
    tolerance = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_entropy_loss = 0
        epoch_cl_loss = 0
        iterator = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        iterator.set_description(f"Fold {fold} Training Epoch {epoch + 1}/{num_epochs}")

        cluster_assign = [] # num of samples in each cluster
        groups = []
        times = []
        events = []
        sample_ids = []
        predictions = []
        for batch_idx, batch in iterator:
            image = batch['image'].to(device)
            group = batch['group'].to(device)
            time = batch['time'].to(device)
            event = batch['event'].to(device)
            lambda_ = batch['lambda'].to(device)
            factors = batch['clinical'].to(device) if 'clinical' in batch else None
            sample_id = batch['sample_id']

            optimizer.zero_grad()
            output = model(image)
            # print(output[0:5]) # print the first 5 outputs for debugging
            loss_ent = entropy_loss(output)
            # loss_ent = torch.tensor(0.0).to(device) # disable entropy loss
            loss_cl = contrastive_loss(output, time, event, lambda_, group, softmax=True)

            loss = loss_cl + loss_ent
            # loss = loss_cl + loss_ent

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_cl_loss += loss_cl.item()
            epoch_entropy_loss += loss_ent.item()

            iterator.set_postfix({
                'epoch': epoch + 1,
                'loss': f'{loss.item():.3f}',
                'loss_cl': f'{loss_cl.item():.3f}',
                'loss_ent': f'{loss_ent.item():.3f}',
                'lr': optimizer.param_groups[0]['lr'],
            })

            pred_probs = torch.softmax(output / args.temperature, dim=1).detach().cpu().numpy()
            predictions.append(pred_probs) # prediction probabilities
            cluster_assign.append(output.argmax(dim=1).detach().cpu().numpy()) # cluster assignment
            groups.append(group.detach().cpu().numpy()) # group labels
            times.append(time.detach().cpu().numpy()) # time to event
            events.append(event.detach().cpu().numpy()) # event
            sample_ids.append(sample_id.cpu().numpy()) # sample ids

            if logger is not None:
                logger[f'train/fold{fold}/batch_loss'].log(loss.item())

        if scheduler is not None:
            scheduler.step()

        epoch_loss /= len(train_loader)
        epoch_cl_loss /= len(train_loader)
        epoch_entropy_loss /= len(train_loader)
        if logger is not None:
            logger[f'train/fold{fold}/epoch_loss'].log(epoch_loss)

        # num of samples in cluster 0 and 1
        cluster_assign = np.concatenate(cluster_assign)
        predictions = np.concatenate(predictions)
        groups = np.concatenate(groups)
        times = np.concatenate(times)
        events = np.concatenate(events)
        sample_ids = np.concatenate(sample_ids)

        n_cluster1 = np.sum(cluster_assign == 0)
        n_cluster2 = np.sum(cluster_assign == 1)
        logging.info(f'Train fold{fold} epoch {epoch + 1}/{num_epochs}: '
                     f'loss: {epoch_loss:.3f}, cl_loss: {epoch_cl_loss:.3f}, '
                     f'entropy_loss: {epoch_entropy_loss:.3f}, '
                     f'n_cluster1: {n_cluster1}, n_cluster2: {n_cluster2}')

        # plot survival curves
        if (epoch + 1) % args.log_every_n_epoch == 0:
            # time and event for cluster 0 and 1 in cases
            log_survival_plot(logger, times, events, cluster_assign, groups, stage='train', epoch=epoch, fold=fold,
                              snapshot_path=snapshot_path)

        # save checkpoints
        if (epoch + 1) % args.save_every == 0:
            if isinstance(model, nn.DataParallel):
                save_dict = {
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.module.state_dict(),
                }
            else:
                save_dict = {
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                }
            torch.save(save_dict, os.path.join(snapshot_path, f'model_epoch_{epoch + 1}.pth'))

        # evaluate after warmup
        if scheduler is None or (epoch + 1) > args.warmup_epochs:
            test_loss = evaluate(args, logger, model, test_loader, device, epoch, fold, snapshot_path)
            if logger is not None:
                logger[f'test/fold{fold}/epoch_loss'].log(test_loss)


@torch.no_grad()
def evaluate(args: Namespace, logger, model, test_loader, device: torch.device,
             epoch, fold=1, snapshot_path=None):
    model.eval()
    test_loss = 0
    test_entropy_loss = 0
    test_cl_loss = 0

    contrastive_loss = ContrastiveLoss(w1=args.w1, w2=args.w2, minp=args.minp, temperature=args.temperature)
    entropy_loss = EntropyLoss(weight=args.w3, temperature=args.temperature)

    iterator = tqdm(enumerate(test_loader), total=len(test_loader), ncols=125)
    iterator.set_description(f"Fold {fold} Evaluating Epoch {epoch + 1}/{args.max_epochs}")
    cluster_assign = [] # num of samples in each cluster
    groups = []
    times = []
    events = []
    sample_ids = []
    predictions = []
    for batch_idx, batch in iterator:
        image = batch['image'].to(device)
        group = batch['group'].to(device)
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

        loss_cl = contrastive_loss(output, time, event, lambda_, group, softmax=True)
        loss_ent = entropy_loss(output)

        loss = loss_cl + loss_ent

        test_loss += loss.item()
        test_cl_loss += loss_cl.item()
        test_entropy_loss += loss_ent.item()
        iterator.set_postfix({
            'loss': f'{loss.item():.3f}',
            'loss_cl': f'{loss_cl.item():.3f}',
            'loss_ent': f'{loss_ent.item():.3f}',
        })

        if logger is not None:
            logger[f'test/fold{fold}/batch_loss'].log(loss.item())

        cluster_assign.append(output.argmax(dim=1).detach().cpu().numpy())
        pred_probs = torch.softmax(output / args.temperature, dim=1).detach().cpu().numpy()
        predictions.append(pred_probs)
        groups.append(group.detach().cpu().numpy())
        times.append(time.detach().cpu().numpy())
        events.append(event.detach().cpu().numpy())
        sample_ids.append(sample_id.cpu().numpy())

    test_loss /= len(test_loader)
    test_cl_loss /= len(test_loader)
    test_entropy_loss /= len(test_loader)

    cluster_assign = np.concatenate(cluster_assign)
    predictions = np.concatenate(predictions)
    groups = np.concatenate(groups)
    times = np.concatenate(times)
    events = np.concatenate(events)
    sample_ids = np.concatenate(sample_ids)
    # save the results
    if snapshot_path is not None:
        df = pd.DataFrame({
            'Eid': sample_ids,
            'group': groups,
            'cluster_assign': cluster_assign,
            'pred_cluster0': predictions[:, 0],
            'pred_cluster1': predictions[:, 1],
            'time': times,
            'event': events
        })
        df.to_csv(os.path.join(snapshot_path, 'clusters', f'test_fold{fold}_epoch{epoch + 1}.csv'), index=False)
    # num of samples in cluster 0 and 1
    n_cluster1 = np.sum(cluster_assign == 0)
    n_cluster2 = np.sum(cluster_assign == 1)

    # plot survival curves
    if (epoch + 1) % args.log_every_n_epoch == 0:
        log_survival_plot(logger, times, events, cluster_assign, groups, stage='test', epoch=epoch, fold=fold,
                          snapshot_path=snapshot_path)

    logging.info(f'test fold{fold} epoch {epoch + 1} loss: {test_loss:.3f}, '
                 f'cl_loss: {test_cl_loss:.3f}, entropy_loss: {test_entropy_loss:.3f}'
                 f'n_cluster1: {n_cluster1}, n_cluster2: {n_cluster2}')

    return test_loss


def log_survival_plot(logger, times, events, cluster_assign, groups, stage='train',
                      epoch=0, fold=1, snapshot_path=None):

    # plot survival curves for cases and controls
    controls = groups == 0
    cases = groups == 1
    cluster0 = cluster_assign == 0
    cluster1 = cluster_assign == 1

    time_case0 = times[cases & cluster0]
    event_case0 = events[cases & cluster0]
    time_case1 = times[cases & cluster1]
    event_case1 = events[cases & cluster1]
    if snapshot_path is not None:
        save_path = os.path.join(snapshot_path, 'figs')
    else:
        save_path = None

    survival_plot(logger, time_case0, event_case0, time_case1, event_case1, group='case', stage=stage,
                  fold=fold, epoch=epoch + 1, save_path=save_path)
    # time and event for cluster 0 and 1 in controls
    time_control0 = times[controls & cluster0]
    event_control0 = events[controls & cluster0]
    time_control1 = times[controls & cluster1]
    event_control1 = events[controls & cluster1]

    survival_plot(logger, time_control0, event_control0, time_control1, event_control1,
                  group='control', stage=stage,
                  fold=fold, epoch=epoch + 1, save_path=save_path)
