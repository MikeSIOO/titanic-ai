import itertools
from copy import deepcopy

import pandas as pd

from nn_utils.configs import ExperimentConfig
from nn_utils.cv_runner import cv_nn


# поиск по сетке параметров
def grid_search_nn(
    X_train,
    y_train,
    base_cfg: ExperimentConfig,
    param_grid: dict,
    model_factory, # фактори создания нейросети
    columns_for_nn,
    n_splits=5,
    seed=42,
) -> 'pd.DataFrame':
    """
    param_grid — плоский словарь с точечной нотацией, формирующий сетку поиска
    {
        'mlp.hidden_sizes': [[32], [64], [32, 16]],
        'mlp.dropout': [0.0, 0.3],
        'training.lr': [1e-3, 1e-2],
        'training.optimizer': ['sgd', 'adam'],
    }
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    rows = []

    for combo in combos:
        # формирование конфига для текущего эксперимента из сетки параметров
        cfg = deepcopy(base_cfg)
        for key, value in zip(keys, combo):
            obj = cfg
            parts = key.split('.')
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], value)

        # обучение нейросети
        result = cv_nn(X_train, y_train, cfg, model_factory,
                       columns_for_nn, n_splits, seed)

        # накапливание истории
        rows.append({
            'params': dict(zip(keys, combo)),
            'mean_acc': result['mean_acc'],
            'std_acc': result['std_acc'],
        })
        print(f"{dict(zip(keys, combo))} -> "
              f"{result['mean_acc']:.4f} ± {result['std_acc']:.4f}")

    return pd.DataFrame(rows).sort_values('mean_acc', ascending=False)