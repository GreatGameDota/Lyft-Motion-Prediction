import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path

from Config import cfg

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, transform2=None, mode='train'):
        self.dataset = dataset
        self.transform = transform
        self.transform2 = transform2
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
          return int(len(self.dataset) * .001)
          # return 5000
          # return 100
        else:
          return 100
          # return 10000
        # return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
          idx = random.randint(0, len(self.dataset)-1)
        batch = self.dataset[idx]

        img = batch['image'].transpose(1, 2, 0)
        # img = self.dataset.rasterizer.to_rgb(img)
        # img = cv2.bitwise_not(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = np.rollaxis(img, 0, )

        if self.transform is not None:
          img = self.transform(image=img)['image']
        img = np.rollaxis(img, -1, 0)

        target_avail = batch["target_availabilities"]
        targets = batch['target_positions']
        matrix = batch["world_to_image"]
        centroid = batch["centroid"]
        if self.mode == 'val':
          agent_ids = batch["track_id"]
          timestamps = batch["timestamp"]
          world_from_agents = batch["world_from_agent"]
          return [img, targets, target_avail, matrix, centroid, agent_ids, timestamps, world_from_agents]
        else:
          return [img, targets, target_avail, matrix, centroid]

# from: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/195936
class MyTrainDataset:
    def __init__(self, cfg, dm):
        self.cfg = cfg
        self.dm = dm
        self.has_init = False
    def initialize(self, worker_id):
        # print('initialize called with worker_id', worker_id)
        from l5kit.data import ChunkedDataset
        from l5kit.dataset import AgentDataset #, EgoDataset
        from l5kit.rasterization import build_rasterizer
        rasterizer = build_rasterizer(self.cfg, self.dm)
        train_cfg = self.cfg["train_data_loader"]
        train_zarr = ChunkedDataset(self.dm.require(train_cfg["key"])).open(cached=False)  # try to turn off cache
        self.dataset = AgentDataset(self.cfg, train_zarr, rasterizer)
        self.has_init = True
    def reset(self):
        self.dataset = None
        self.has_init = False
    def __len__(self):
        # note you have to figure out the actual length beforehand since once the rasterizer and/or AgentDataset been constructed, you cannot pickle it anymore! So we can't compute the size from the real dataset. However, DataLoader require the len to determine the sampling.
        return int(22496709 * .01)
    def __getitem__(self, index):
        index = random.randint(0, 22496709 - 1)
        return self.dataset[index]
        
class MyValDataset:
    def __init__(self, cfg, dm):
        self.cfg = cfg
        self.dm = dm
        self.has_init = False
    def initialize(self, worker_id):
        # print('initialize called with worker_id', worker_id)
        from l5kit.data import ChunkedDataset
        from l5kit.dataset import AgentDataset #, EgoDataset
        from l5kit.rasterization import build_rasterizer
        rasterizer = build_rasterizer(self.cfg, self.dm)
        
        eval_cfg = self.cfg["val_data_loader"]
        eval_base_path = 'data/validate_chopped_100/'
        eval_zarr_path = str(Path(eval_base_path) / Path(self.dm.require(eval_cfg["key"])).name)
        eval_mask_path = str(Path(eval_base_path) / "mask.npz")
        eval_gt_path = str(Path(eval_base_path) / "gt.csv")

        eval_zarr = ChunkedDataset(eval_zarr_path).open(cached=False)  # try to turn off cache
        eval_mask = np.load(eval_mask_path)["arr_0"]
        self.dataset = AgentDataset(self.cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
        self.has_init = True
    def reset(self):
        self.dataset = None
        self.has_init = False
    def __len__(self):
        # note you have to figure out the actual length beforehand since once the rasterizer and/or AgentDataset been constructed, you cannot pickle it anymore! So we can't compute the size from the real dataset. However, DataLoader require the len to determine the sampling.
        return 94694
        # return 10000
    def __getitem__(self, index):
        return self.dataset[index]
        
def my_dataset_worker_init_func(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.initialize(worker_id)

def get_loader(dataset, dm, batch_size=128, workers=0, shuffle=True, transform=None, mode='train'):
    # dataset = ImageDataset(dataset, transform, mode=mode)
    # loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    if mode == 'train':
      dataset = MyTrainDataset(cfg, dm)
      loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                            # persistent_workers=True,
                            worker_init_fn=my_dataset_worker_init_func)
    elif mode == 'val':
      dataset = MyValDataset(cfg, dm)
      loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                            # persistent_workers=True,
                            worker_init_fn=my_dataset_worker_init_func)
    return loader
