import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import random

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
          # return 100
          return 10000
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
        
def get_loader(dataset, batch_size=128, workers=0, shuffle=True, transform=None, mode='train'):
    dataset = ImageDataset(dataset, transform, mode=mode)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return loader
