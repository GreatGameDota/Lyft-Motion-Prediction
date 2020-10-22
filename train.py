import random
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

# from utils import *

class config:
    epochs = 10
    batch_size = 4
    regression = False
    num_classes = 6 - 1
    IMAGE_PATH = 'data/train/'
    lr = 1e-4
    # lr = 3e-4
    N = 36
    sz = 256
    mean = [1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
    std = [0.36357649, 0.49984502, 0.40477625],
    seed = 69420
    mixup = 0
    cutmix = 0
    accumulation_steps = 1
    single_fold = 0
    apex = True


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
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
    
    length = len(os.listdir('./logs/')) - 1
    log_name = "./logs/log-%i.log"%(length)

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

        # print('Loading previous training...')
        # state = torch.load('model.pth')
        # model.load_state_dict(state['model_state'])
        # best = state['kaggle']
        # best2 = state['loss']
        # print(f'Loaded model with kaggle score: {best}, loss: {best2}')
        # optimizer.load_state_dict(state['opt_state'])
        # scheduler.load_state_dict(state['scheduler_state'])
        # early_epoch = state['epoch'] + 1
        # print(f'Beginning at epoch {early_epoch}')
        # print('')

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
            
            # ###################################################################
            # ############## VALIDATION #########################################
            # ###################################################################

            model.eval()
            loss = 0
            
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

            if epoch > 0:
                history2['acc'].plot()
                plt.savefig('epoch%03d_%i_acc.png'%(epoch+1,fold))
                plt.clf()
            
            if loss < best2:
                best2 = loss
                print('Saving best model... (loss)')
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'kaggle': final_score,
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                }, 'model-1_%i.pth'%(fold))
            
            if final_score > best:
                best = final_score
                print('Saving best model... (acc)')
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'kaggle': final_score,
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                }, 'model_%i.pth'%(fold))

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--checkpoint' default=None, stye=str)
#     return parser.parse_args()

if __name__ == '__main__':
    main()
