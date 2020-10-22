import torch

def get_optimizer(model, lr=0.001, lookahead=False):
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    return base_opt