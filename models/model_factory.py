import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from pytorchcv.model_provider import get_model

from torch.nn.parameter import Parameter


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

class Head(torch.nn.Module):
  def __init__(self, in_f, out_f, dropout=True):
    super(Head, self).__init__()
    
    self.f = nn.Flatten()

    self.l = nn.Linear(in_f, 512)
    # self.m = Mish()
    
    self.d = nn.Dropout(0.5)
    self.dropout = dropout

    # self.o = nn.Linear(512, out_f)
    self.o = nn.Linear(in_f, out_f)

    self.b1 = nn.BatchNorm1d(in_f)
    self.b2 = nn.BatchNorm1d(512)

  def forward(self, x):
    x = self.f(x)
    # x = self.b1(x)
    if self.dropout:
      x = self.d(x)

    # x = self.l(x)
    # x = self.m(x)
    # x = self.b2(x)
    # x = self.d(x)

    out = self.o(x)
    return out

class FCN(torch.nn.Module):
  def __init__(self, base, in_f, classes=5, dropout=True):
    super(FCN, self).__init__()
    self.base = base
    self.h1 = Head(in_f, classes, dropout=dropout)
  
  def forward(self, x):
    x = self.base(x)
    return self.h1(x)

def load_model(name, checkpoint_path=None, pretrained=False):
    model = get_model(name, pretrained=True)
    # model = timm.create_model(name, pretrained=True)
    try:
      features = list(model.children())[-1].in_features
    except:
      features = list(model.children())[-1][-1].in_features
    model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
    # model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    model[0].final_pool = nn.Sequential(GeM())
    # model[-1] = nn.Sequential(GeM())
    # model = nn.Sequential(*list(m.children())[:-2])
    model = FCN(model, features, 6, dropout=True)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    
    return model
