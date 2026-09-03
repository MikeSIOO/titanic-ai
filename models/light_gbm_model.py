import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier

import constants


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.LIGHT_GBM_MODEL['ENABLED']:
        columns = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
        if constants.LIGHT_GBM_MODEL['FIND_BEST_PARAMS']:
            param_grid = {
                'n_estimators': [10, 20, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [15, 31, 63, 127],
                'subsample': [0.7, 0.8, 0.9, 1.0],
            }
            grid = GridSearchCV(
                estimator=LGBMClassifier(random_state=constants.RANDOM_STATE, verbosity=-1),
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE),
                scoring='accuracy',
                n_jobs=-1,
            )
            grid.fit(X_train[columns], y_train)
            print(f'Лучшее значение accuracy для бустинга (LightGBM) с параметрами {grid.best_params_} CV: {grid.best_score_:.4f}')
        else:
            model = LGBMClassifier(
                random_state=constants.RANDOM_STATE,
                verbosity=-1,
                n_estimators=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['n_estimators'],
                learning_rate=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['learning_rate'],
                max_depth=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['max_depth'],
                num_leaves=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['num_leaves'],
                subsample=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['subsample'],
            )
            model_cv_results = cross_validate(
                model, X_train[columns], y_train,
                cv=StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE),
                scoring=['accuracy']
            )
            print(f'Accuracy для бустинга (LightGBM) CV: {model_cv_results['test_accuracy'].mean():.4f}, CV STD: {model_cv_results['test_accuracy'].std():.4f}')
    else:
        print('Бустинг (LightGBM) не обучено. Флаг LIGHT_GBM_MODEL["ENABLED"]')
