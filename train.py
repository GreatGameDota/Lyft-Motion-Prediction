import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
import sklearn.metrics
import gc
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F

from apex import amp

import warnings
warnings.filterwarnings("ignore")

from dataloader import get_loader
from models import load_model
from optimizers import get_optimizer
from schedulers import get_scheduler
from transforms import get_transform
from losses import get_criterion

from Config import config

# from utils import *

import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_model(model, val_loader, criterion1, epoch, scheduler, history, log_name=None):
    model.eval()
    loss = 0.
    
    preds_1 = []
    tars_1 = []
    with torch.no_grad():
        # t = tqdm(val_loader)
        for img_batch, y_batch in val_loader:
            img_batch = img_batch.cuda().float()
            y_batch = y_batch.cuda().long()

            o1 = model(img_batch)

            l1  = criterion1(o1, y_batch)
            loss += l1

            for j in range(len(o1)):
                preds_1.append(torch.argmax(F.softmax(o1[j]), -1))
            for i in y_batch:
                tars_1.append(i[0].data.cpu().numpy())
    
    preds_1 = [p.data.cpu().numpy() for p in preds_1]
    preds_1 = np.array(preds_1).T.reshape(-1)

    final_score =  sklearn.metrics.recall_score(
        tars_1, preds_1, average='macro')
    
    loss /= len(val_loader)
    
    if history2 is not None:
        history2.loc[epoch, 'val_loss'] = loss.cpu().numpy()
        history2.loc[epoch, 'acc'] = final_score
    
    if scheduler is not None:
        scheduler.step(final_score)

    print('Dev loss: %.4f, Kaggle: %.4f' % (loss, final_score))
    
    with open(log_name, 'a') as f:
        f.write('XXXXXXXXXXXXXX-- CYCLE INTER: %i --XXXXXXXXXXXXXXXXXXX\n'%(epoch+1))
        f.write('val epoch: %i\n'%(epoch+1))
        f.write('val loss: %.4f  val acc: %.4f\n'%(loss,acc))
        f.write('val QWK: %.4f\n'%(final_score))
        f.write('\n')

    return preds_1, tars_1, loss, final_score

def main():
    if not os.path.isdir('data/lyft-scenes/'):
        os.system('python download.py')
    seed_everything(config.seed)
#     args = parse_args()

    train_df = pd.read_csv('data/train_with_folds.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data Loaders
    # df_train, df_val = train_test_split(train_df, test_size=0.2, random_state=2021)

    # train_transform = get_transform(128)
    train_transform = A.Compose([
                                A.CoarseDropout(max_holes=1, max_width=64, max_height=64, p=0.9),
                                A.ShiftScaleRotate(rotate_limit=5, p=0.9),
                                A.Normalize(mean=config.mean, std=config.std, always_apply=True)
    ])
    val_transform = A.Compose([
                            A.Normalize(mean=config.mean, std=config.std, always_apply=True)
    ])

    folds = [0, 1, 2, 3, 4]
    
    log_name = f"../drive/My Drive/logs/log-{len(os.listdir('../drive/My Drive/logs/'))}.log"

    # Loop over folds
    for fld in range(1):
        fold = config.single_fold
        print('Train fold: %i'%(fold+1))
        with open(log_name, 'a') as f:
            f.write('Train Fold %i\n\n'%(fold+1))

        train_loader = get_loader(train_df, config.IMAGE_PATH, folds=[
                                  i for i in folds if i != fold], batch_size=config.batch_size, workers=4, shuffle=True, transform=train_transform)
        val_loader = get_loader(train_df, config.IMAGE_PATH, folds=[fold], batch_size=config.batch_size, workers=4, shuffle=False, transform=val_transform)

        # Build Model
        model = load_model('seresnext50_32x4d', pretrained=True)
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
        scheduler = get_scheduler(optimizer, train_loader=train_loader, epochs=n_epochs, batch_size=config.batch_size)

        for epoch in range(n_epochs-early_epoch):
            epoch += early_epoch
            torch.cuda.empty_cache()
            gc.collect()

            # ###################################################################
            # ############## TRAINING ###########################################
            # ###################################################################

            model.train()
            total_loss = 0
            
            t = tqdm(train_loader)
            for batch_idx, (img_batch, y_batch) in enumerate(t):
                img_batch = img_batch.cuda().float()
                y_batch = y_batch.cuda().long()
                
                # optimizer.zero_grad()
                
                rand = np.random.rand()
                if rand < config.mixup:
                    pass
                    # images, targets = mixup(img_batch, y_batch, 0.4)
                    # output1 = model(images)
                    # l1 = mixup_criterion(output1, targets)
                elif rand < config.cutmix:
                    pass
                    # images, targets = cutmix(img_batch, y_batch, 0.4)
                    # output1 = model(images)
                    # l1 = cutmix_criterion(output1, targets)
                else:
                    output1 = model(img_batch)
                    loss = criterion1(output1, y_batch) / config.accumulation_steps

                total_loss += loss.data.cpu().numpy() * config.accumulation_steps
                t.set_description('Epoch %i/%i, LR: %6f, Loss: %.4f' % (epoch+1,n_epochs,
                    optimizer.state_dict()['param_groups'][0]['lr'], total_loss/(batch_idx+1)))

                if history is not None:
                    history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
                    history.loc[epoch + batch_idx / len(train_loader), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                    
                # scaler.scale(loss).backward()
                # loss.backward()
                if config.apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if (batch_idx+1) % config.accumulation_steps == 0:
                    # scaler.step(optimizer)
                    optimizer.step()
                    # scaler.update()
                    optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step(epoch)

            #### VALIDATION ####

            pred, tars, loss, kaggle = evaluate_model(model, val_loader, criterion1, epoch, scheduler=scheduler, history=history2, log_name=log_name)
            
            if kaggle > best:
                best = kaggle
                print(f'Saving best model... (metric)')
                torch.save({
                    'model_state': model.state_dict(),
                }, f'../drive/My Drive/Models/model1-fld{fold+1}.pth')
                with open(log_name, 'a') as f:
                    f.write('Saving Best model...\n\n')
            else:
                with open(log_name, 'a') as f:
                    f.write('\n')
        
        model = create_model('resnet18', path=f'../drive/My Drive/Models/model1-fld{fold+1}.pth')
        model.cuda()
        pred, tars, loss, kaggle = evaluate_model(model, val_loader, criterion1, 0, scheduler=scheduler, history=history2, log_name=log_name)

if __name__ == '__main__':
    main()
