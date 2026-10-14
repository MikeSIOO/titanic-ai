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
    if not constants.TWO_LAYER_SIMPLE_MODEL['ENABLED']:
        print('Нейросеть с двумя слоями не обучена. Флаг TWO_LAYER_SIMPLE_MODEL["ENABLED"]')
        return

    if constants.TWO_LAYER_SIMPLE_MODEL['FIND_BEST_PARAMS']:
        base_cfg = ExperimentConfig(
            mlp=MLPConfig(hidden_sizes=[32]),
            training=TrainingConfig(lr=1e-2, optimizer='adam', num_epochs=20),
        )
        # чтобы не искать сразу по полной сетке выберем 3 основных параметра
        param_grid = {
            'mlp.hidden_sizes': [[8], [16], [32], [64], [128]],
            'training.lr': [1e-3, 1e-2, 1e-1],
            'training.optimizer': ['sgd', 'adam', 'adamw']
        }
        # теперь можно найти остальные
        param_grid = { # best 'mlp.hidden_sizes': [32], 'training.lr': 0.01, 'training.optimizer': 'adam'
            'mlp.hidden_sizes': [[16], [32], [64]], # дополнительно оставим выбор размера внутреннего слоя вблизи оптимального с предыдущего шага (32)
            'mlp.activations': [['relu'], ['tanh'], ['sigmoid'], ['leaky_relu']],
            'training.weight_decay': [1e-3, 1e-2, 1e-1, 0],
            'training.batch_size': [32, 64, 128]
        }
        results = grid_search_nn(
            X_train,
            y_train,
            base_cfg,
            param_grid,
            model_factory=build_mlp,
            columns_for_nn=COLUMNS_FOR_NN,
        )
        print(results.head(5).to_string())
    else:
        base_cfg = ExperimentConfig(
            mlp=MLPConfig(
                hidden_sizes=constants.TWO_LAYER_SIMPLE_MODEL['BEST_PARAMS']['hidden_sizes'],
                activations=constants.TWO_LAYER_SIMPLE_MODEL['BEST_PARAMS']['activations'],
            ),
            training=TrainingConfig(
                lr=constants.TWO_LAYER_SIMPLE_MODEL['BEST_PARAMS']['lr'],
                weight_decay=constants.TWO_LAYER_SIMPLE_MODEL['BEST_PARAMS']['weight_decay'],
                optimizer=constants.TWO_LAYER_SIMPLE_MODEL['BEST_PARAMS']['optimizer'],
                batch_size=constants.TWO_LAYER_SIMPLE_MODEL['BEST_PARAMS']['batch_size'],
                num_epochs=constants.TWO_LAYER_SIMPLE_MODEL['NUM_EPOCHES'],
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
