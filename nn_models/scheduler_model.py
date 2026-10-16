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
    if not constants.SCHEDULER_MODEL['ENABLED']:
        print('Нейросеть не обучена с шедулером. Флаг SCHEDULER_MODEL["ENABLED"]')
        return

    if constants.SCHEDULER_MODEL['FIND_BEST_PARAMS']:
        base_cfg = ExperimentConfig(
            mlp=MLPConfig(
                hidden_sizes=[64, 32],
                activations=['tanh', 'tanh'],
            ),
            training=TrainingConfig(
                num_epochs=constants.TWO_LAYER_SIMPLE_MODEL['NUM_EPOCHES'],
                weight_decay=1e-3,
            ),
        )
        # теперь можно найти остальные
        param_grid = {
            'training.scheduler': ['none', 'step', 'cosine'],
            'training.num_epochs': [10, 25, 50, 100],
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
                hidden_sizes=[64, 32],
                activations=['tanh', 'tanh'],
            ),
            training=TrainingConfig(
                weight_decay=1e-3,
                scheduler=constants.SCHEDULER_MODEL['BEST_PARAMS']['scheduler'],
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
