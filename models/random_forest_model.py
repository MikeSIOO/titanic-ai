import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV

import constants


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.RANDOM_FOREST_MODEL['ENABLED']:
        columns_for_decision_tree = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
        if constants.RANDOM_FOREST_MODEL['FIND_BEST_PARAMS']:
            param_grid = {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy', 'log_loss'],
            }
            grid = GridSearchCV(
                estimator=RandomForestClassifier(random_state=constants.RANDOM_STATE),
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE),
                scoring='accuracy')
            grid.fit(X_train[columns_for_decision_tree], y_train)
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
            decision_tree_model_cv_results = cross_validate(
                decision_tree_model, X_train[columns_for_decision_tree], y_train,
                cv=StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE),
                scoring=['accuracy']
            )
            print(f'Accuracy для случайного леса CV: {decision_tree_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {decision_tree_model_cv_results['test_accuracy'].std():.4f}')

            # Визуализация важности признаков
            decision_tree_model.fit(X_train[columns_for_decision_tree], y_train)
            importances = decision_tree_model.feature_importances_
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
