import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import constants
from data.feature_engineering import DataTransformer
from data.select_columns_for_model import ColumnSelector


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.XG_BOOST_MODEL['ENABLED']:
        columns = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
        cv = StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE)

        if constants.XG_BOOST_MODEL['FIND_BEST_PARAMS']:
            param_grid = {
                'model__max_depth': [3, 5, 7, 9],
                'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
                'model__n_estimators': [50, 100, 200], # данных мало, поэтому при больших значения сильное переобучение
                'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            pipeline = Pipeline([
                ('transformer', DataTransformer()),
                ('selector', ColumnSelector(columns)),
                ('model', XGBClassifier(random_state=constants.RANDOM_STATE, verbosity=0)),
            ])
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            print(f'Лучшее значение accuracy для бустинга (XGBoost) с параметрами {grid.best_params_} CV: {grid.best_score_:.4f}')
        else:
            model = XGBClassifier(
                random_state=constants.RANDOM_STATE,
                verbosity=0,
                max_depth=constants.XG_BOOST_MODEL['BEST_PARAMS']['max_depth'],
                learning_rate=constants.XG_BOOST_MODEL['BEST_PARAMS']['learning_rate'],
                n_estimators=constants.XG_BOOST_MODEL['BEST_PARAMS']['n_estimators'],
                subsample=constants.XG_BOOST_MODEL['BEST_PARAMS']['subsample']
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
            print(f'Accuracy для бустинга (XGBoost) CV: {model_cv_results['test_accuracy'].mean():.4f}, CV STD: {model_cv_results['test_accuracy'].std():.4f}')
    else:
        print('Бустинг (XGBoost) не обучено. Флаг XG_BOOST_MODEL["ENABLED"]')
