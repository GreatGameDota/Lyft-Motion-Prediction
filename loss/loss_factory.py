import torch
import torch.nn as nn
import torch.nn.functional as F

def criterion1(pred1, targets):
  l1 = F.cross_entropy(pred1, targets)
  return l1

def get_criterion():
    return criterion1