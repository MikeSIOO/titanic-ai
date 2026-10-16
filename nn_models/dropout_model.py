import numpy as np

import constants
from nn_utils.configs import ExperimentConfig, MLPConfig, TrainingConfig
from nn_utils.cv_runner import cv_nn
from nn_utils.grid_search_runner import grid_search_nn
from nn_utils.model_factory import build_mlp
from matplotlib import pyplot as plt

COLUMNS_FOR_NN = [
    'Embarked_Q', 'Embarked_S', 'Sex_male',
    'Pclass_norm', 'Age_norm', 'SibSp_norm', 'Parch_norm',
    'Fare_norm', 'Family_size_norm',
]

def run_model(X_train, y_train):
    if not constants.DROPOUT_MODEL['ENABLED']:
        print('Нейросеть с дропаут. Флаг DROPOUT_MODEL["ENABLED"]')
        return

    if constants.DROPOUT_MODEL['FIND_BEST_PARAMS']:
        base_cfg = ExperimentConfig(
            mlp=MLPConfig(),
            training=TrainingConfig(weight_decay=1e-3),
        )
        # чтобы не искать сразу по полной сетке выберем основные параметры
        param_grid = {
            'mlp.hidden_sizes': [
                [8, 8],
                [16, 16],
                [32, 32],
                [64, 64],
                [128, 128],

                [8, 16],
                [16, 32],
                [32, 64],
                [64, 128],
                [128, 256],

                [8, 4],
                [16, 8],
                [32, 16],
                [64, 32],
                [128, 64]
            ]
        }
        # теперь можно найти остальные
        param_grid = {
            'mlp.hidden_sizes': [[32, 16], [8, 4], [64, 32]], # дополнительно оставим выбор размера внутреннего слоя вблизи оптимального с предыдущего шага
            'mlp.activations': [
                ['relu', 'relu'], ['tanh', 'relu'], ['leaky_relu', 'relu'],
                ['relu', 'tanh'], ['tanh', 'tanh'], ['leaky_relu', 'tanh'],
                ['relu', 'sigmoid'], ['tanh', 'sigmoid'], ['leaky_relu', 'sigmoid'],
                ['relu', 'leaky_relu'], ['tanh', 'leaky_relu'], ['leaky_relu', 'leaky_relu'],
            ],
            'mlp.dropout': [0.0, 0.1, 0.2, 0.3]
        }
        results = grid_search_nn(
            X_train,
            y_train,
            base_cfg,
            param_grid,
            model_factory=build_mlp,
            columns_for_nn=COLUMNS_FOR_NN,
        )
        print(results.head(20).to_string())
    else:
        base_cfg = ExperimentConfig(
            mlp=MLPConfig(
                hidden_sizes=constants.DROPOUT_MODEL['BEST_PARAMS']['hidden_sizes'],
                activations=constants.DROPOUT_MODEL['BEST_PARAMS']['activations'],
                dropout=constants.DROPOUT_MODEL['BEST_PARAMS']['dropout'],
            ),
            training=TrainingConfig(
                num_epochs=constants.DROPOUT_MODEL['NUM_EPOCHES'],
            ),
        )
        result = cv_nn(
            X_train, y_train, base_cfg,
            model_factory=build_mlp, columns_for_nn=COLUMNS_FOR_NN,
            return_histories=True,
        )
        print(f"CV: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
        plt.plot(np.mean([loss_history[0] for loss_history in result['histories']], axis=0), label='loss')
        plt.show()

        plt.plot(np.mean([loss_history[1] for loss_history in result['histories']], axis=0), label='train accuracy')
        plt.plot(np.mean([loss_history[2] for loss_history in result['histories']], axis=0), label='val accuracy')
        plt.legend()
        plt.show()
