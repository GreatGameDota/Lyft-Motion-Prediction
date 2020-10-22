import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import *

def criterion1(targets, pred, confidence, target_avail):
  return pytorch_neg_multi_log_likelihood_batch(targets, pred, confidence, target_avail)

def get_criterion():
    return criterion1