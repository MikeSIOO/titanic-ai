import pandas as pd
from catboost import CatBoostClassifier, cv, Pool
from matplotlib import pyplot as plt

import constants


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.CAT_BOOST_MODEL['ENABLED']:
        columns = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Family_size', 'Sex', 'Embarked']
        cat_features = ['Sex', 'Embarked']
        if constants.CAT_BOOST_MODEL['FIND_BEST_PARAMS']:
            model = CatBoostClassifier(
                cat_features=cat_features,
                random_seed=constants.RANDOM_STATE,
                eval_metric='Accuracy',
                verbose=False,
                early_stopping_rounds=10
            )
            param_grid = {
                'depth': [3, 5, 7, 9],
                'l2_leaf_reg': [1, 3, 5, 7, 10],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'iterations': [10, 20, 50, 100], # данных мало, поэтому при больших значения сильное переобучение
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'grow_policy': ['SymmetricTree', 'Depthwise']
            }
            grid = model.grid_search(
                param_grid=param_grid,
                X=X_train[columns],
                y=y_train,
                cv=constants.N_SPLITS,
                shuffle=True,
                stratified=True,
                partition_random_seed=constants.RANDOM_STATE,
            )
            print(f'Лучшее значение accuracy для бустинга (CatBoost) с параметрами {grid["params"]} CV: {max(grid["cv_results"]["test-Accuracy-mean"]):.4f}')

            plt.plot(grid["cv_results"]['train-Accuracy-mean'], label='train accuracy')
            plt.plot(grid["cv_results"]['test-Accuracy-mean'], label='test accuracy')
            plt.legend()
            plt.show()
        else:
            pool = Pool(
                X_train[columns],
                y_train,
                cat_features=cat_features
            )
            params = {
                'loss_function': 'Logloss',
                'random_seed': constants.RANDOM_STATE,
                'eval_metric': 'Accuracy',
                'depth': constants.CAT_BOOST_MODEL['BEST_PARAMS']['depth'],
                'l2_leaf_reg': constants.CAT_BOOST_MODEL['BEST_PARAMS']['l2_leaf_reg'],
                'learning_rate': constants.CAT_BOOST_MODEL['BEST_PARAMS']['learning_rate'],
                'iterations': constants.CAT_BOOST_MODEL['BEST_PARAMS']['iterations'],
                'subsample': constants.CAT_BOOST_MODEL['BEST_PARAMS']['subsample'],
                'grow_policy': constants.CAT_BOOST_MODEL['BEST_PARAMS']['grow_policy'],
            }
            cv_data = cv(
                pool=pool,
                params=params,
                fold_count=constants.N_SPLITS,
                shuffle=True,
                stratified=True,
                partition_random_seed=constants.RANDOM_STATE,
                verbose=False
            )
            print(f'Accuracy для бустинга (CatBoost) CV: {cv_data['test-Accuracy-mean'].iloc[-1]:.4f}, CV STD: {cv_data['test-Accuracy-std'].iloc[-1]:.4f}')

            plt.plot(cv_data['train-Accuracy-mean'], label='train accuracy')
            plt.plot(cv_data['test-Accuracy-mean'], label='test accuracy')
            plt.legend()
            plt.show()
    else:
        print('Бустинг (CatBoost) не обучено. Флаг CAT_BOOST_MODEL["ENABLED"]')
