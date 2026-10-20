from itertools import product

import constants
from final_nn.nn_utils import search_grid, cross_validate_nn

def run_model(X_train, y_train):
    if not constants.FINAL_MODEL['ENABLED']:
        print('Финальная модель не обучена. Флаг FINAL_MODEL["ENABLED"]')
        return

    if constants.FINAL_MODEL['FIND_BEST_PARAMS']:
        # find_best_hidden(X_train, y_train) # 64 -> 0.825993
        # find_two_best_hidden(X_train, y_train) # [64, 16] -> 0.826012 (остальные хуже)
        # find_three_best_hidden(X_train, y_train) # [64, 16, 8] -> 0.823771 (хуже)
        # find_best_batchnorm(X_train, y_train) # False -> 0.826012
        # find_best_dropout(X_train, y_train) # 0.1 -> 0.829370
        # find_best_activation(X_train, y_train) # relu -> 0.829370
        # find_best_optimizer(X_train, y_train) # optimizer_name=adam, lr=0.010, weight_decay=0.001 -> 0.829370
        # find_best_scheduler(X_train, y_train) # none -> 0.829370
        # find_best_batch_size(X_train, y_train) # 64 -> 0.829370
        find_best_num_epochs(X_train, y_train) # 30 -> 0.828259
    else: # Финальная CV: 0.8485 ± 0.0139
        final = cross_validate_nn(
            X_train, y_train,
            hidden_sizes=constants.FINAL_MODEL['BEST_PARAMS']['hidden_sizes'],
            batchnorm=constants.FINAL_MODEL['BEST_PARAMS']['batchnorm'],
            dropout=constants.FINAL_MODEL['BEST_PARAMS']['dropout'],
            activation=constants.FINAL_MODEL['BEST_PARAMS']['activation'],
            optimizer_name=constants.FINAL_MODEL['BEST_PARAMS']['optimizer_name'],
            lr=constants.FINAL_MODEL['BEST_PARAMS']['lr'],
            weight_decay=constants.FINAL_MODEL['BEST_PARAMS']['weight_decay'],
            scheduler_name=constants.FINAL_MODEL['BEST_PARAMS']['scheduler_name'],
            batch_size=constants.FINAL_MODEL['BEST_PARAMS']['batch_size'],
            num_epochs=constants.FINAL_MODEL['BEST_PARAMS']['num_epochs'],
            return_histories=True,
        )
        print(f"Финальная CV: {final['mean_acc']:.4f} ± {final['std_acc']:.4f}")

        _plot_histories(final['histories'])

def find_best_hidden(X_train, y_train):
    # поиск лучшего размера скрытого слоя
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[4], [8], [16], [32], [64], [128]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_two_best_hidden(X_train, y_train):
    # поиск лучшего размера 2 скрытых слоев
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 8], [64, 16], [64, 32], [64, 64], [64, 128]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_three_best_hidden(X_train, y_train):
    # поиск лучшего размера 3 скрытых слоев (размер первых двух - лучший вариант из предыдущего эксперимента)
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 16, 8], [64, 16, 16], [64, 16, 32], [64, 16, 64], [64, 16, 128]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_best_batchnorm(X_train, y_train):
    # поиск batchnorm
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 16]],
            'batchnorm': [False, True],
            'dropout': [0.0],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_best_dropout(X_train, y_train):
    # поиск dropout
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 16]],
            'batchnorm': [False],
            'dropout': [0.0, 1e-3, 1e-2, 0.1, 0.2],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_best_activation(X_train, y_train):
    # поиск activation
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 16]],
            'batchnorm': [False],
            'dropout': [0.1],
            'activation': ['relu', 'tanh', 'leaky_relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_best_optimizer(X_train, y_train):
    # поиск optimizer
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 16]],
            'batchnorm': [False],
            'dropout': [0.1],
            'activation': ['relu'],
            'optimizer_name': ['sgd', 'adam'],
            'lr': [1e-3, 1e-2, 1e-1],
            'weight_decay': [0, 1e-3, 1e-2, 1e-1],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_best_scheduler(X_train, y_train):
    # поиск scheduler
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 16]],
            'batchnorm': [False],
            'dropout': [0.1],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none', 'step', 'cosine'],
            'batch_size': [64],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_best_batch_size(X_train, y_train):
    # поиск batch_size
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 16]],
            'batchnorm': [False],
            'dropout': [0.1],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [16, 32, 64, 128, 256],
            'num_epochs': [20],
        },
    )
    print(results.to_string())

def find_best_num_epochs(X_train, y_train):
    # поиск num_epochs
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[64, 16]],
            'batchnorm': [False],
            'dropout': [0.1],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [32],
            'num_epochs': [10, 20, 25, 30, 50, 70],
        },
    )
    print(results.to_string())


def _plot_histories(histories):
    import matplotlib.pyplot as plt
    import numpy as np

    loss_mean = np.mean([h[0] for h in histories], axis=0)
    train_mean = np.mean([h[1] for h in histories], axis=0)
    val_mean = np.mean([h[2] for h in histories], axis=0)

    plt.plot(loss_mean, label='loss')
    plt.xlabel('Эпоха')
    plt.legend()
    plt.show()

    plt.plot(train_mean, label='train accuracy')
    plt.plot(val_mean, label='val accuracy')
    plt.xlabel('Эпоха')
    plt.legend()
    plt.show()