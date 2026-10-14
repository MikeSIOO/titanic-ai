import torch
from torch import nn

from nn_utils.configs import TrainingConfig


# фабрика создания оптимизаторов
def build_optimizer(model: nn.Module, cfg: TrainingConfig):
    params = model.parameters()
    if cfg.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'adam':
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")