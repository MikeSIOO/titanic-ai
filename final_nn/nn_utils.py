import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

from data.feature_engineering import DataTransformer

# Список колонок, которые подаются в нейросеть (после DataTransformer)
COLUMNS_FOR_NN = [
    'Embarked_Q', 'Embarked_S', 'Sex_male',
    'Pclass_norm', 'Age_norm', 'SibSp_norm', 'Parch_norm',
    'Fare_norm', 'Family_size_norm',
]


def build_mlp(in_features, hidden_sizes, batchnorm=False, dropout=0.0, activation='relu'):
    """
    Собирает простой MLP: Linear -> Act -> (Dropout) -> ... -> Linear(..., 1).

    hidden_sizes: список размеров скрытых слоёв, например [16] или [32, 16].
    batchnorm: нужно ли добавлять батчнорм
    dropout: значение dropout
    activation: 'relu' | 'tanh' | 'leaky_relu'.
    """
    act_cls = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'leaky_relu': nn.LeakyReLU,
    }[activation]

    layers = []
    prev = in_features
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev, hidden))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def make_loaders(X_tr, y_tr, X_val, y_val, batch_size=64):
    """DataFrame -> два DataLoader'а (train, val)."""
    X_tr_np = X_tr[COLUMNS_FOR_NN].to_numpy(dtype=np.float32, copy=True)
    X_val_np = X_val[COLUMNS_FOR_NN].to_numpy(dtype=np.float32, copy=True)
    y_tr_np = y_tr.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
    y_val_np = y_val.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

    train_ds = TensorDataset(torch.from_numpy(X_tr_np), torch.from_numpy(y_tr_np))
    val_ds = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_np))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def train_one_fold(X_tr, y_tr, X_val, y_val,
                   hidden_sizes, batchnorm, dropout, activation,
                   lr, weight_decay, batch_size, num_epochs,
                   optimizer_name='adam',
                   scheduler_name='none',
                   device=None):
    """
    Обучает одну модель на одной паре (train, val) и возвращает:
    - best_val_acc: лучшая accuracy на валидации за все эпохи
    - histories: (loss_history, train_history, val_history)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = make_loaders(
        X_tr, y_tr, X_val, y_val, batch_size=batch_size
    )

    model = build_mlp(
        in_features=len(COLUMNS_FOR_NN),
        hidden_sizes=hidden_sizes,
        batchnorm=batchnorm,
        dropout=dropout,
        activation=activation,
    ).to(device)

    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)

    scheduler = build_scheduler(scheduler_name, optimizer, num_epochs)

    loss_fn = nn.BCEWithLogitsLoss()

    loss_history, train_history, val_history = [], [], []

    for epoch in range(num_epochs):
        # --- train ---
        model.train()
        loss_accum = 0.0
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()
            correct += ((torch.sigmoid(pred) > 0.5).float() == y).sum().item()
            total += y.shape[0]

        loss_history.append(loss_accum / len(train_loader))
        train_history.append(correct / total)

        # --- val ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                correct += ((torch.sigmoid(pred) > 0.5).float() == y).sum().item()
                total += y.shape[0]
        val_acc = correct / total
        val_history.append(val_acc)

        if scheduler is not None:
            scheduler.step()

    return val_acc, (loss_history, train_history, val_history)


def cross_validate_nn(X_train, y_train,
                      hidden_sizes, batchnorm, dropout, activation,
                      lr, weight_decay, batch_size, num_epochs,
                      optimizer_name='adam',
                      scheduler_name='none',
                      n_splits=5, seed=42,
                      return_histories=False):
    """
    Честная кросс-валидация: DataTransformer обучается на каждом фолде отдельно.
    Возвращает dict с mean_acc, std_acc и (опционально) histories.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs = []
    all_histories = []

    for tr_idx, val_idx in skf.split(X_train, y_train):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        X_tr_raw = X_train.iloc[tr_idx]
        X_val_raw = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        transformer = DataTransformer()
        X_tr = transformer.fit_transform(X_tr_raw)
        X_val = transformer.transform(X_val_raw)

        best_val_acc, histories = train_one_fold(
            X_tr, y_tr, X_val, y_val,
            hidden_sizes=hidden_sizes,
            batchnorm=batchnorm,
            dropout=dropout,
            activation=activation,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_epochs=num_epochs,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
        )
        fold_accs.append(best_val_acc)
        all_histories.append(histories)

    result = {
        'mean_acc': float(np.mean(fold_accs)),
        'std_acc': float(np.std(fold_accs)),
        'fold_accs': fold_accs,
    }
    if return_histories:
        result['histories'] = all_histories
    return result


def search_grid(X_train, y_train, param_grid, n_splits=5, seed=42):
    """
    Простой перебор по сетке.
    param_grid — dict с плоскими ключами: hidden_sizes, dropout, activation, lr, ...
    Пример:
        param_grid = {
            'hidden_sizes': [[16], [32]],
            'dropout': [0.0, 0.2],
            'lr': [1e-3, 1e-2],
        }
    Возвращает DataFrame отсортированный по mean_acc.
    """
    import itertools

    keys = list(param_grid.keys())
    rows = []

    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo))
        result = cross_validate_nn(
            X_train, y_train,
            n_splits=n_splits, seed=seed,
            **params,
        )
        row = {**params, 'mean_acc': result['mean_acc'], 'std_acc': result['std_acc']}
        rows.append(row)
        print(f"{params} -> {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")

    return pd.DataFrame(rows).sort_values('mean_acc', ascending=False)

def build_optimizer(model, optimizer_name, lr, weight_decay):
    if optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

def build_scheduler(scheduler_name, optimizer, num_epochs):
    if scheduler_name == 'none':
        return None
    elif scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        raise ValueError(f'Unknown scheduler: {scheduler_name}')

def nn_fit_and_predict(X_tr, y_tr, X_val,
                       columns_for_nn,
                       hidden_sizes, batchnorm, dropout, activation,
                       optimizer_name, lr, weight_decay, scheduler_name,
                       batch_size, num_epochs,
                       device=None, seed=42):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- DataLoader только для train ---
    X_tr_np = X_tr[columns_for_nn].to_numpy(dtype=np.float32, copy=True)
    y_tr_np = y_tr.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
    train_ds = TensorDataset(torch.from_numpy(X_tr_np), torch.from_numpy(y_tr_np))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # --- Модель, оптимизатор, scheduler ---
    model = build_mlp(
        in_features=len(COLUMNS_FOR_NN),
        hidden_sizes=hidden_sizes,
        batchnorm=batchnorm,
        dropout=dropout,
        activation=activation
    ).to(device)
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = build_scheduler(scheduler_name, optimizer, num_epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    # --- Обучение ---
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # --- Предсказание ---
    model.eval()
    X_val_np = X_val[columns_for_nn].to_numpy(dtype=np.float32, copy=True)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val_np).to(device))
        probas = torch.sigmoid(logits).cpu().numpy().ravel()

    return probas
