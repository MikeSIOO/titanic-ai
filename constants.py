SHOULD_DOWNLOAD_DATA = False  # Нужно ли скачать датасет

SHOULD_PRINT_DATA_ANALYTICS = False  # Нужно ли проводить анализ данных

N_SPLITS = 5  # число фолдов для кросс валидации
RANDOM_STATE = 42  # сид рандомизатора

BASE_MODEL = {  # параметры, относящиеся к обучению базовой модели
    'ENABLED': False  # нужно ли запускать обучение
}

KNN_MODEL = {  # параметры, относящиеся к обучению модели ближайших соседей
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {  # значения лучших параметров
        'n_neighbors': 21
    },
}

LOGISTIC_REGRESSION_MODEL = {  # параметры, относящиеся к обучению модели линейной регрессии
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {  # значения лучших параметров
        'lr': 0.1
    },
}

DECISION_TREE_MODEL = {  # параметры, относящиеся к обучению модели решающего дерева
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {  # значения лучших параметров
        'criterion': 'gini',
        'max_depth': 3,
        'min_samples_leaf': 5,
        'min_samples_split': 2,
    },
}

RANDOM_FOREST_MODEL = {  # параметры, относящиеся к обучению модели случайного леса
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {  # значения лучших параметров
        'criterion': 'entropy',
        'max_depth': None,
        'min_samples_leaf': 2,
        'min_samples_split': 2,
        'n_estimators': 200
    },
}

CAT_BOOST_MODEL = {  # параметры, относящиеся к обучению модели градиентного бустинга через библиотеку CatBoost
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {  # значения лучших параметров
        'grow_policy': 'Depthwise',
        'subsample': 0.7,
        'depth': 7,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'iterations': 10
    },
}

LIGHT_GBM_MODEL = {  # параметры, относящиеся к обучению модели градиентного бустинга через библиотеку LightGBM
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {  # значения лучших параметров
        'learning_rate': 0.1,
        'max_depth': -1,
        'n_estimators': 50,
        'num_leaves': 15,
        'subsample': 0.7,
    },
}

XG_BOOST_MODEL = {  # параметры, относящиеся к обучению модели градиентного бустинга через библиотеку XGBoost
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {  # значения лучших параметров
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 100,
        'subsample': 0.8,
    },
}

TWO_LAYER_SIMPLE_MODEL = {  # параметры, относящиеся к обучению нейросети с 2 слоями
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'NUM_EPOCHES': 20,
    'BEST_PARAMS': {  # значения лучших параметров
        'hidden_sizes': [16],
        'activations': ['leaky_relu'],
        'lr': 1e-2,
        'optimizer': 'adam',
        'weight_decay': 1e-2,
        'batch_size': 64
    },
}

MULTIPLE_LAYERS_MODEL = {  # параметры, относящиеся к обучению нейросети
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'NUM_EPOCHES': 25,
    'BEST_PARAMS': {  # значения лучших параметров
        'hidden_sizes': [64, 32],
        'activations': ['tanh', 'tanh'],
        'weight_decay': 1e-3
    },
}

BATCHNORM_MODEL = {  # параметры, относящиеся к обучению нейросети
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'NUM_EPOCHES': 25,
    'BEST_PARAMS': {  # значения лучших параметров
        'hidden_sizes': [32, 16],
        'activations': ['relu', 'leaky_relu'],
        'batchnorm': True,
        'weight_decay': 1e-3
    },
}

DROPOUT_MODEL = {  # параметры, относящиеся к обучению нейросети
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'NUM_EPOCHES': 25,
    'BEST_PARAMS': {  # значения лучших параметров
        'hidden_sizes': [32, 16],
        'activations': ['leaky_relu', 'sigmoid'],
        'dropout': 0.1,
    },
}

SCHEDULER_MODEL = {  # параметры, относящиеся к обучению нейросети
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'NUM_EPOCHES': 50,
    'BEST_PARAMS': {  # значения лучших параметров
        'scheduler': 'cosine'
    },
}

FINAL_MODEL = {  # параметры, относящиеся к обучению нейросети
    'ENABLED': False,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {
        'hidden_sizes': [32, 64, 32],
        'batchnorm': False,
        'dropout': 0.0,
        'activation': 'leaky_relu',
        'optimizer_name': 'adam',
        'lr': 1e-2,
        'weight_decay': 1e-3,
        'scheduler_name': 'none',
        'batch_size': 64,
        'num_epochs': 50
    }
}
