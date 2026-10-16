from itertools import product

import constants
from final_nn.nn_utils import search_grid, cross_validate_nn

def run_model(X_train, y_train):
    if not constants.FINAL_MODEL['ENABLED']:
        print('Финальная модель не обучена. Флаг FINAL_MODEL["ENABLED"]')
        return

    if constants.FINAL_MODEL['FIND_BEST_PARAMS']:
        find_best_hidden(X_train, y_train) # 8 -> 0.840638, 32 -> 0.840613, 64 -> 0.838353, 128 -> 0.838347, 16 -> 0.832754, 4 -> 0.824926
        find_two_best_hidden(X_train, y_train) # [32, 64] -> 0.845094, [32, 8] -> 0.840613, [32, 32] -> 0.840600 (остальные хуже 8)
        find_three_best_hidden(X_train, y_train) # [32, 64, 32] -> 0.845113 (остальные хуже [32, 64])
        find_four_best_hidden(X_train, y_train) # результаты хуже, чем с 3 скрытыми слоями
        find_best_batchnorm(X_train, y_train) # результаты хуже, чем в предыдущих экспериментах
        find_best_dropout(X_train, y_train) # результаты хуже, чем в предыдущих экспериментах
        find_best_activation(X_train, y_train) # leaky_relu -> 0.847342, relu -> 0.845113, tanh -> 0.840619
        find_best_optimizer(X_train, y_train) # optimizer_name=adam, lr=0.010, weight_decay=0.001 -> 0.847342
        find_best_scheduler(X_train, y_train) # результаты хуже, чем в предыдущих экспериментах
        find_best_batch_size(X_train, y_train) # 64 -> 0.847342
        find_best_num_epochs(X_train, y_train) # 70 -> 0.848465, 50 -> 0.848465, 100 -> 0.848465
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
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_two_best_hidden(X_train, y_train):
    # поиск лучшего размера 2 скрытых слоев (размер первого - два лучших варианта из предыдущего эксперимента)
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [
                [8, 8], [32, 8],
                [8, 16], [32, 16],
                [8, 32], [32, 32],
                [8, 64], [32, 64],
                [8, 128], [32, 128],
            ],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_three_best_hidden(X_train, y_train):
    # поиск лучшего размера 3 скрытых слоев (размер первых двух - лучший вариант из предыдущего эксперимента)
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 8], [32, 64, 16], [32, 64, 32], [32, 64, 64], [32, 64, 128]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_four_best_hidden(X_train, y_train):
    # поиск лучшего размера 4 скрытых слоев (размер первых трех - лучший вариант из предыдущего эксперимента)
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 32, 8], [32, 64, 32, 16], [32, 64, 32, 32], [32, 64, 32, 64], [32, 64, 32, 128]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_best_batchnorm(X_train, y_train):
    # поиск batchnorm
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 32]],
            'batchnorm': [False, True],
            'dropout': [0.0],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_best_dropout(X_train, y_train):
    # поиск dropout
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 32]],
            'batchnorm': [False],
            'dropout': [0.0, 1e-3, 1e-2, 0.1, 0.2],
            'activation': ['relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_best_activation(X_train, y_train):
    # поиск activation
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 32]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['relu', 'tanh', 'leaky_relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_best_optimizer(X_train, y_train):
    # поиск optimizer
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 32]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['leaky_relu'],
            'optimizer_name': ['sgd', 'adam'],
            'lr': [1e-3, 1e-2, 1e-1],
            'weight_decay': [0, 1e-3, 1e-2, 1e-1],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_best_scheduler(X_train, y_train):
    # поиск scheduler
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 32]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['leaky_relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none', 'step', 'cosine'],
            'batch_size': [64],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_best_batch_size(X_train, y_train):
    # поиск batch_size
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 32]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['leaky_relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [16, 32, 64, 128, 256],
            'num_epochs': [30],
        },
    )
    print(results.to_string())

def find_best_num_epochs(X_train, y_train):
    # поиск num_epochs
    results = search_grid(
        X_train, y_train,
        param_grid={
            'hidden_sizes': [[32, 64, 32]],
            'batchnorm': [False],
            'dropout': [0.0],
            'activation': ['leaky_relu'],
            'optimizer_name': ['adam'],
            'lr': [1e-2],
            'weight_decay': [1e-3],
            'scheduler_name': ['none'],
            'batch_size': [64],
            'num_epochs': [10, 20, 30, 50, 70, 100],
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