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
    'ENABLED': True,  # нужно ли запускать обучение
    'FIND_BEST_PARAMS': False,  # нужно ли искать лучшие параметры
    'BEST_PARAMS': {  # значения лучших параметров
        'criterion': 'entropy',
        'max_depth': None,
        'min_samples_leaf': 2,
        'min_samples_split': 2,
        'n_estimators': 200
    },
}
