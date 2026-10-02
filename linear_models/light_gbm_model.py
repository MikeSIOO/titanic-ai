import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

import constants
from data.feature_engineering import DataTransformer
from data.select_columns_for_model import ColumnSelector


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.LIGHT_GBM_MODEL['ENABLED']:
        columns = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
        cv = StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE)

        if constants.LIGHT_GBM_MODEL['FIND_BEST_PARAMS']:
            param_grid = {
                'model__n_estimators': [10, 20, 50, 100],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7, -1],
                'model__num_leaves': [15, 31, 63, 127],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
            }
            pipeline = Pipeline([
                ('transformer', DataTransformer()),
                ('selector', ColumnSelector(columns)),
                ('model', LGBMClassifier(random_state=constants.RANDOM_STATE, verbosity=-1)),
            ])
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
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
            pipeline = Pipeline([
                ('transformer', DataTransformer()),
                ('selector', ColumnSelector(columns)),
                ('model', model),
            ])
            model_cv_results = cross_validate(
                pipeline, X_train, y_train,
                cv=cv,
                scoring=['accuracy']
            )
            print(f'Accuracy для бустинга (LightGBM) CV: {model_cv_results['test_accuracy'].mean():.4f}, CV STD: {model_cv_results['test_accuracy'].std():.4f}')
    else:
        print('Бустинг (LightGBM) не обучено. Флаг LIGHT_GBM_MODEL["ENABLED"]')
