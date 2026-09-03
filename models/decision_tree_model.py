import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

import constants


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.DECISION_TREE_MODEL['ENABLED']:
        columns_for_decision_tree = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
        if constants.DECISION_TREE_MODEL['FIND_BEST_PARAMS']:
            param_grid = {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5, 10],
            }
            grid = GridSearchCV(
                estimator=DecisionTreeClassifier(random_state=constants.RANDOM_STATE),
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE),
                scoring='accuracy',
                n_jobs=-1
            )
            grid.fit(X_train[columns_for_decision_tree], y_train)
            print(f'Лучшее значение accuracy для решающего дерева с параметрами {grid.best_params_} CV: {grid.best_score_:.4f}')
        else:
            decision_tree_model = DecisionTreeClassifier(
                criterion=constants.DECISION_TREE_MODEL['BEST_PARAMS']['criterion'],
                max_depth=constants.DECISION_TREE_MODEL['BEST_PARAMS']['max_depth'],
                min_samples_leaf=constants.DECISION_TREE_MODEL['BEST_PARAMS']['min_samples_leaf'],
                min_samples_split=constants.DECISION_TREE_MODEL['BEST_PARAMS']['min_samples_split']
            )
            decision_tree_model_cv_results = cross_validate(
                decision_tree_model, X_train[columns_for_decision_tree], y_train,
                cv=StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE),
                scoring=['accuracy']
            )
            print(f'Accuracy для решающего дерева CV: {decision_tree_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {decision_tree_model_cv_results['test_accuracy'].std():.4f}')

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
            plt.title('Важность признаков в решающем дереве')
            plt.tight_layout()
            plt.show()

            # Визуализация дерева
            plt.figure(figsize=(15, 10))
            plot_tree(decision_tree_model, feature_names=columns_for_decision_tree, class_names=['died', 'survived'], filled=True, rounded=True)
            plt.show()
    else:
        print('Решающее дерево не обучено. Флаг DECISION_TREE_MODEL["ENABLED"]')
