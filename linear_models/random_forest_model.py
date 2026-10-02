import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

import constants
from data.feature_engineering import DataTransformer
from data.select_columns_for_model import ColumnSelector


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.RANDOM_FOREST_MODEL['ENABLED']:
        columns_for_decision_tree = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
        cv = StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE)

        if constants.RANDOM_FOREST_MODEL['FIND_BEST_PARAMS']:
            param_grid = {
                'model__n_estimators': [50, 100, 200, 500],
                'model__max_depth': [3, 5, 7, 10, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 5, 10],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
            }
            decision_tree_model = RandomForestClassifier(random_state=constants.RANDOM_STATE)
            pipeline = Pipeline([
                ('transformer', DataTransformer()),
                ('selector', ColumnSelector(columns_for_decision_tree)),
                ('model', decision_tree_model),
            ])
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            print(f'Лучшее значение accuracy для случайного леса с параметрами {grid.best_params_} CV: {grid.best_score_:.4f}')
        else:
            decision_tree_model = RandomForestClassifier(
                n_estimators=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['n_estimators'],
                criterion=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['criterion'],
                max_depth=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['max_depth'],
                min_samples_leaf=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['min_samples_leaf'],
                min_samples_split=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['min_samples_split'],
                random_state=constants.RANDOM_STATE
            )

            pipeline = Pipeline([
                ('transformer', DataTransformer()),
                ('selector', ColumnSelector(columns_for_decision_tree)),
                ('model', decision_tree_model),
            ])

            decision_tree_model_cv_results = cross_validate(
                pipeline, X_train, y_train,
                cv=cv,
                scoring=['accuracy']
            )
            print(f'Accuracy для случайного леса CV: {decision_tree_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {decision_tree_model_cv_results['test_accuracy'].std():.4f}')

            # Визуализация важности признаков
            best_model = RandomForestClassifier(
                n_estimators=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['n_estimators'],
                criterion=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['criterion'],
                max_depth=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['max_depth'],
                min_samples_leaf=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['min_samples_leaf'],
                min_samples_split=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['min_samples_split'],
                random_state=constants.RANDOM_STATE
            )
            transformer = DataTransformer()
            X_train = transformer.fit_transform(X_train)
            best_model.fit(X_train[columns_for_decision_tree], y_train)
            importances = best_model.feature_importances_
            feature_names = columns_for_decision_tree
            sorted_idx = np.argsort(importances)[::-1]
            sorted_names = [feature_names[i] for i in sorted_idx]
            sorted_importances = importances[sorted_idx]
            plt.figure(figsize=(10, 6))
            colors = ['red' if c < 0 else 'green' for c in importances]
            plt.barh(sorted_names, sorted_importances, color=colors)
            plt.xlabel('Важность признака')
            plt.title('Важность признаков в случайном лесе')
            plt.tight_layout()
            plt.show()
    else:
        print('Случайноый лес не обучен. Флаг RANDOM_FOREST_MODEL["ENABLED"]')
