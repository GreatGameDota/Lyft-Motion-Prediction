import argparse
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import glob
import math
from tqdm import tqdm,trange
from pathlib import Path
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from torch.utils.data.dataset import Subset

from torch import nn, optim
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.evaluation import create_chopped_dataset, write_pred_csv, compute_metrics_csv, read_gt_csv
from l5kit.geometry import transform_points

import warnings
warnings.filterwarnings("ignore")

from dataloader import get_loader
from models import load_model
from optimizers import get_optimizer
from schedulers import get_scheduler
from transform import get_transform
from loss import get_criterion

from Config import config, cfg

from utilities import *

import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_model(model, val_loader, criterion1, epoch, eval_gt_path, scheduler, history, log_name=None):
    model.eval()
    loss = 0.
    metric_ = 0.
    
    preds_1 = []
    confids = []
    tars_1 = []
    ids, times = [], []
    with torch.no_grad():
        t = tqdm(val_loader)
        for batch in t:
        # for img_batch, targets, y_avail, matrix, centroid, agent_ids, timestamps, world_from_agents in t:
        # tr_it = iter(val_loader)
        # t = trange(math.ceil(cfg['val_params']['steps'] / cfg["val_data_loader"]["batch_size"]))
        # for batch_idx in t:
        #     try:
        #         batch = next(tr_it)
        #     except StopIteration:
        #         tr_it = iter(val_loader)
        #         batch = next(tr_it)
            img_batch = batch['image'].cuda().float()
            y_avail = batch["target_availabilities"].cuda().float()
            targets = batch['target_positions'].cuda().float()
            matrix = batch["world_to_image"].cuda().float()
            centroid = batch["centroid"][:,None,:].cuda().float()
            agent_ids = batch["track_id"].cuda()
            timestamps = batch["timestamp"].cuda()
            world_from_agents = batch["world_from_agent"].cuda().float()

            bs,tl,_ = targets.shape
            assert tl == cfg["model_params"]["future_num_frames"]

            pred, confid = model(img_batch)

            # print(y_batch, pred)
            l1 = criterion1(targets, pred, confid, y_avail)
            loss += l1

            for idx in range(len(pred)):
              for mode in range(3):
                  pred[idx, mode, :, :] = transform_points(pred[idx, mode, :, :], world_from_agents[idx]) - centroid[idx][:2]
            # metric_ += metric(o1, y_batch)

            for j in range(len(pred)):
              preds_1.append(pred[j].data.cpu().numpy())
              confids.append(confid[j].data.cpu().numpy())
              ids.append(agent_ids[j].data.cpu().numpy())
              times.append(timestamps[j].data.cpu().numpy())
            for i in targets:
              tars_1.append(i.data.cpu().numpy())
    
    # preds_1 = np.array(preds_1)
    pred_path = f"pred.csv"
    # print(np.array(times))
    write_pred_csv(pred_path,
               timestamps=np.array(times),
               track_ids=np.array(ids),
               coords=np.array(preds_1),
               confs=np.array(confids)
    )
    
    gt = pd.read_csv('data/validate_chopped_100/gt2.csv')
    gt = gt.loc[:len(preds_1)-1]
    gt.to_csv('data/validate_chopped_100/gt.csv', index=False)

    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
    final_score = metrics['neg_multi_log_likelihood']
    
    loss = final_score
    
    if history is not None:
      history.loc[epoch, 'val_loss'] = loss
      # history.loc[epoch, 'metric'] = final_score
    
    if scheduler is not None:
      scheduler.step(loss)

    # print(f'Dev loss: %.4f, Metric: {final_score}, Metric2: {final_score2}'%(loss))
    # print(f'Dev loss: %.4f, Metric: {final_score}'%(loss))
    
    with open(log_name, 'a') as f:
      f.write(f'val loss: {loss}\n')
      # f.write(f'val Metric: {final_score}\n')

    return preds_1, tars_1, loss, ids, times

def main():
    if not os.path.isdir('data/aerial_map/'):
        os.system('python download.py')
    seed_everything(config.seed)
#     args = parse_args()

    INPUT_DIR = './'
    os.environ["L5KIT_DATA_FOLDER"] = INPUT_DIR
    dm = LocalDataManager(None)

    # if not os.path.isfile('data/validate_chopped_100/gt2.csv'):
        # os.system("cp 'data/validate_chopped_100/gt.csv' 'data/validate_chopped_100/gt2.csv'")

    ###########################################
    ############## LOAD DATA ##################
    ###########################################

    train_cfg = cfg['train_data_loader']

    # # Rasterizer
    # rasterizer = build_rasterizer(cfg, dm)

    # # Train dataset/dataloader
    # train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open(cached=False)
    # train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    # # train_dataloader = DataLoader(train_dataset,
    # #                             shuffle=train_cfg["shuffle"],
    # #                             batch_size=train_cfg["batch_size"],
    # #                             num_workers=train_cfg["num_workers"])

    # print(len(train_dataset))
    # gc.collect()

    # idxs = np.arange(len(train_dataset))
    # np.random.shuffle(idxs)
    # sampler = SubsetRandomSampler( idxs[:int(len(train_dataset) * .005)] )
    # train_loader = DataLoader(train_dataset,
    #                             shuffle=False,
    #                             batch_size=train_cfg["batch_size"],
    #                             num_workers=train_cfg["num_workers"],
    #                             sampler=sampler)

    num_frames_to_chop = 100
    MIN_FUTURE_STEPS = 10
    eval_cfg = cfg['val_data_loader']

    if not os.path.isdir('data/validate_chopped_100/'):
        eval_base_path = create_chopped_dataset(dm.require(eval_cfg['key']), cfg["raster_params"]["filter_agents_threshold"], 
                                        num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)
    else:
        eval_base_path = 'data/validate_chopped_100/'

    eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
    eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    # eval_zarr = ChunkedDataset(eval_zarr_path).open(cached=False)
    # eval_mask = np.load(eval_mask_path)["arr_0"]
    # # ===== INIT DATASET AND LOAD MASK
    # eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)

    # # eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
    # #                             num_workers=eval_cfg["num_workers"])
    # print(len(eval_dataset))
    # gc.collect()
    
    # idxs = np.arange(len(eval_dataset))
    # eval_dataset = Subset( eval_dataset, idxs[:cfg['val_params']['steps']] )
    # val_loader = DataLoader(eval_dataset, shuffle=False, batch_size=eval_cfg["batch_size"], 
    #                             num_workers=eval_cfg["num_workers"], sampler=None)

    ###############################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data Loaders

    train_transform = A.Compose([
    ])
    val_transform = A.Compose([
                            # A.Normalize(mean=config.mean, std=config.std, always_apply=True)
    ])
    
    log_name = f"./logs/log-{len(os.listdir('./logs/'))}.log"

    # Loop over folds
    for fld in range(1):
        fold = config.single_fold
        print('Train fold: %i'%(fold+1))
        with open(log_name, 'a') as f:
            f.write('Train Fold %i\n\n'%(fold+1))
        
        # train_loader = get_loader(train_dataset, batch_size=config.batch_size, workers=0, shuffle=True, transform=None)
        # val_loader = get_loader(eval_dataset, batch_size=config.batch_size, workers=0, shuffle=False, transform=None, mode='val')
    
        train_loader, train_dataset = get_loader(None, dm, batch_size=train_cfg['batch_size'], workers=train_cfg['num_workers'], shuffle=train_cfg['shuffle'], transform=None)
        val_loader, val_dataset = get_loader(None, dm, batch_size=eval_cfg['batch_size'], workers=eval_cfg['num_workers'], shuffle=eval_cfg['shuffle'], transform=None, mode='val')

        scaler = amp.GradScaler()

        # Build Model
        model = load_model('resnet18')
        # model = load_model('resnet18', path='model1-fld1.pth')
        model = model.cuda()

        # Optimizer
        optimizer = get_optimizer(model, lr=config.lr)

        # Apex
        if config.apex:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O1', verbosity=0)

        # Loss
        criterion1 = get_criterion()

        # Training
        history = pd.DataFrame()
        history2 = pd.DataFrame()

        torch.cuda.empty_cache()
        gc.collect()

        best = 0
        best2 = 1e10
        n_epochs = config.epochs
        early_epoch = 0
        
        # Scheduler
        scheduler = get_scheduler(optimizer, train_loader=train_loader, train_dataset=train_dataset, epochs=n_epochs, batch_size=train_cfg['batch_size'])
        updates_per_epoch = math.ceil(len(train_dataset) / train_cfg['batch_size'])

        for epoch in range(n_epochs-early_epoch):
            epoch += early_epoch
            torch.cuda.empty_cache()
            gc.collect()

            # ###################################################################
            # ############## TRAINING ###########################################
            # ###################################################################

            model.train()
            total_loss = 0
            global_step = epoch * updates_per_epoch
            
            t = tqdm(train_loader)
            for batch_idx, batch in enumerate(t):
            # for batch_idx, (img_batch, targets, y_avail, matrix, centroid) in enumerate(t):
            #     img_batch = img_batch.cuda().float()
            #     targets = targets.cuda().float()
            #     y_avail = y_avail.cuda().float()
            #     matrix = matrix.cuda().float()
            #     centroid = centroid[:,None,:].cuda().float()
            # for batch_idx, batch in enumerate(t):
            # tr_it = iter(train_loader)
            # t = trange(int(len(train_dataset) * .005) // train_cfg["batch_size"])
            # for batch_idx in t:
            #     try:
            #         batch = next(tr_it)
            #     except StopIteration:
            #         tr_it = iter(train_loader)
            #         batch = next(tr_it)
                img_batch = batch['image'].cuda().float()
                y_avail = batch["target_availabilities"].cuda().float()
                targets = batch['target_positions'].cuda().float()
                matrix = batch["world_to_image"].cuda().float()
                centroid = batch["centroid"][:,None,:].cuda().float()
                
                bs,tl,_ = targets.shape
                assert tl == cfg["model_params"]["future_num_frames"]
                
                # try:
                rand = np.random.rand()
                if rand < config.mixup:
                    pass
                elif rand < config.cutmix:
                    pass
                else:
                    if config.scale:
                        with amp.autocast():
                            pred, confid = model(img_batch)
                            loss = criterion1(targets, pred, confid, y_avail) / config.accumulation_steps
                    else:
                        pred, confid = model(img_batch)
                        loss = criterion1(targets, pred, confid, y_avail) / config.accumulation_steps
                
                total_loss += loss.data.cpu().numpy() * config.accumulation_steps
                t.set_description(f'Epoch {epoch+1}/{n_epochs}, LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(batch_idx+1)))

                if history is not None:
                    history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
                    history.loc[epoch + batch_idx / len(train_loader), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                
                if config.scale:
                    scaler.scale(loss).backward()
                elif config.apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                lr_this_step = None
                if (batch_idx+1) % config.accumulation_steps == 0:
                    if config.scale:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

                    # lr_this_step = config.lr * scheduler.get_lr(global_step, 0.03)
                    # for param_group in optimizer.param_groups:
                    #     param_group['lr'] = lr_this_step
                    # global_step += 1
                
                if scheduler is not None:
                   scheduler.step()
                # except AssertionError as e:
                #     print(e)

            ### VALIDATION ####

            pred, tars, loss, ids, times = evaluate_model(model, val_loader, criterion1, epoch, eval_gt_path, scheduler=None, history=history2, log_name=log_name)
            
            if loss < best2:
                best2 = loss
                print(f'Saving best model... (metric)')
                torch.save({
                    'model_state': model.state_dict(),
                }, f'model1-fld{fold+1}.pth')
                with open(log_name, 'a') as f:
                    f.write('Saving Best model...\n\n')
            else:
                with open(log_name, 'a') as f:
                    f.write('\n')
        
        model = load_model('resnet18', path=f'model1-fld{fold+1}.pth')
        model.cuda()
        pred, tars, loss, ids, times = evaluate_model(model, val_loader, criterion1, epoch, eval_gt_path, scheduler=None, history=history2, log_name=log_name)
        
if __name__ == '__main__':
    main()
