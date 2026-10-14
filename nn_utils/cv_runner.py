import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import torch

from data.feature_engineering import DataTransformer
from nn_utils.configs import ExperimentConfig
from nn_utils.model_runner import ModelRunner
from nn_utils.optimizer_factory import build_optimizer
from nn_utils.scheduler_factory import build_scheduler

# создает дата лоадер для тестовой и валидационной выборки (по переданным колонкам) заданного размера батча
def _make_loaders(X_tr, y_tr, X_val, y_val, batch_size, columns):
    X_tr_np = X_tr[columns].astype(np.float32).values
    X_val_np = X_val[columns].astype(np.float32).values
    y_tr_np = y_tr.astype(np.float32).values.reshape(-1, 1)
    y_val_np = y_val.astype(np.float32).values.reshape(-1, 1)

    train_ds = TensorDataset(torch.from_numpy(X_tr_np), torch.from_numpy(y_tr_np))
    val_ds = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_np))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# обучение и валидация нейросети на фолдах
def cv_nn(
    X_train,
    y_train,
    cfg: ExperimentConfig,
    model_factory, # фактори создания нейросети
    columns_for_nn,
    n_splits=5,
    seed=42,
    return_histories=False,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs = []
    histories = []

    for tr_idx, val_idx in skf.split(X_train, y_train):
        # трансформируем данные (обучение трансформера на тестовой части)
        transformer = DataTransformer()
        X_tr = transformer.fit_transform(X_train.iloc[tr_idx])
        X_val = transformer.transform(X_train.iloc[val_idx])
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        # создаем лоадеры для теста и валидации
        train_loader, val_loader = _make_loaders(
            X_tr, y_tr, X_val, y_val,
            cfg.training.batch_size, columns_for_nn,
        )

        # создаем модель и оптимизатор (для каждого фолда создаем заново)
        model = model_factory(in_features=len(columns_for_nn), cfg=cfg.mlp)
        optimizer = build_optimizer(model, cfg.training)
        scheduler = build_scheduler(optimizer, cfg.training)

        # запускаем обучение модели
        runner = ModelRunner(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            num_epochs=cfg.training.num_epochs,
        )
        loss_h, train_h, val_h = runner.train_model()

        # собираем данные по фолду
        fold_accs.append(val_h[-1])
        histories.append((loss_h, train_h, val_h))

    result = {
        'mean_acc': float(np.mean(fold_accs)), # средняя accuracy по всем фолдам
        'std_acc': float(np.std(fold_accs)), # стандартное отклонение accuracy по всем фолдам
        'fold_accs': fold_accs, # список accuracy по всем фолдам
    }
    if return_histories:
        result['histories'] = histories # история изменения лосса, accuracy по трейну и валидации по фолдам
    return result