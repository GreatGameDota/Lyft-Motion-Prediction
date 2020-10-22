import torch

def get_optimizer(model, lr=0.001, lookahead=False):
    base_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    
    return base_opt