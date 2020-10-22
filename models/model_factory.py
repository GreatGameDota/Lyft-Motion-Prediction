import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from pytorchcv.model_provider import get_model

from torch.nn.parameter import Parameter

from Config import config, cfg

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Pooling(nn.Module):
  def __init__(self):
    super(Pooling, self).__init__()
    
    self.p1 = nn.AdaptiveAvgPool2d((1,1))
    self.p2 = nn.AdaptiveMaxPool2d((1,1))

  def forward(self, x):
    x1 = self.p1(x)
    x2 = self.p2(x)
    return (x1+x2) * 0.5

class FCN(torch.nn.Module):
  def __init__(self, base, in_f, num_modes=3, dropout=True):
    super(FCN, self).__init__()
    
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    
    self.base = base
    self.base[0].init_block.conv.conv = nn.Conv2d(
        num_in_channels,
        self.base[0].init_block.conv.conv.out_channels,
        kernel_size=self.base[0].init_block.conv.conv.kernel_size,
        stride=self.base[0].init_block.conv.conv.stride,
        padding=self.base[0].init_block.conv.conv.padding,
        bias=False,
    )
    
    self.future_len = cfg["model_params"]["future_num_frames"]
    num_targets = 2 * self.future_len
    
    self.num_preds = num_targets * num_modes
    self.num_modes = num_modes
    
    self.after_model = nn.Sequential(
        nn.Flatten(),
        # nn.Dropout(0.5)
    )
    self.classification = nn.Sequential(
        # nn.Linear(in_f+100, 1024),
        # nn.BatchNorm1d(1024),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(1024, num_classes),
        nn.Linear(in_f, 4096),
        # nn.Dropout(0.5),
        nn.Linear(4096, self.num_preds + num_modes),
    )
  
  def forward(self, x):
    x = self.base(x)
    x = self.after_model(x)
    x = self.classification(x)

    bs, _ = x.shape
    pred, confidences = torch.split(x, self.num_preds, dim=1)
    pred = pred.view(bs, self.num_modes, self.future_len, 2)
    assert confidences.shape == (bs, self.num_modes)
    # print(confidences)
    confidences = torch.softmax(confidences, dim=1)
    return pred, confidences

def load_model(name, path=None):
    model = get_model(name, pretrained=True)

    try:
      features = list(model.children())[-1].in_features
    except:
      features = list(model.children())[-1][-1].in_features
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    # model[0].init_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1)) # change maxpool in resnet for determinism
    # model[-1] = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    model = FCN(model, features, dropout=True)

    if path:
      print ('loading pretrained model {}'.format(path))
      pretrain = torch.load(path)['model_state']
      model.load_state_dict(pretrain)

    return model
