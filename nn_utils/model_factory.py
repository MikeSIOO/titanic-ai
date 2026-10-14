import torch.nn as nn

from nn_utils.configs import MLPConfig

# словарь функций активации
_ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': lambda: nn.LeakyReLU(0.01),
}

# формирование нейросети
def build_mlp(in_features: int, cfg: MLPConfig) -> nn.Module:
    layers = []
    prev = in_features
    # если передана только 1 активация, применяем ее ко всем слоям
    acts = cfg.activations
    if len(acts) == 1:
        acts = acts * len(cfg.hidden_sizes)

    for hidden, act_name in zip(cfg.hidden_sizes, acts):
        layers.append(nn.Linear(prev, hidden))
        if cfg.batchnorm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(_ACTIVATIONS[act_name]())
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)