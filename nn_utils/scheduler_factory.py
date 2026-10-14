import torch

from nn_utils.configs import TrainingConfig


# фабрика создания шедулеров
def build_scheduler(optimizer, cfg: TrainingConfig):
    if cfg.scheduler == 'none':
        return None
    if cfg.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    if cfg.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs
        )
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")