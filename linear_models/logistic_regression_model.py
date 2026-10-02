import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline

import constants
from data.feature_engineering import DataTransformer
from data.select_columns_for_model import ColumnSelector


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.LOGISTIC_REGRESSION_MODEL['ENABLED']:
        columns_for_logistic_regression = ['Pclass_norm', 'Sex_male', 'Age_bin_norm', 'Fare_bin_norm', 'Fare_by_Pclass_bin_norm', 'Embarked_Q', 'Embarked_S', 'SibSp_norm', 'Parch_norm', 'Family_size_norm']
        cv = StratifiedKFold(n_splits=constants.N_SPLITS, shuffle=True, random_state=constants.RANDOM_STATE)

        if constants.LOGISTIC_REGRESSION_MODEL['FIND_BEST_PARAMS']:
            possible_lr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]
            best_accuracy = None
            best_lr = None
            for lr in possible_lr:
                logistic_regression_model = LogisticRegression(C=lr)
                pipeline = Pipeline([
                    ('transformer', DataTransformer()),
                    ('selector', ColumnSelector(columns_for_logistic_regression)),
                    ('model', logistic_regression_model),
                ])
                logistic_regression_model_cv_results = cross_validate(
                    pipeline, X_train, y_train,
                    cv=cv,
                    scoring=['accuracy']
                )
                print(f'Accuracy для модели линейной регрессии с числом соседей {lr} CV: {logistic_regression_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {logistic_regression_model_cv_results['test_accuracy'].std():.4f}')
                if best_accuracy is None or best_accuracy.mean() < logistic_regression_model_cv_results['test_accuracy'].mean():
                    best_accuracy = logistic_regression_model_cv_results['test_accuracy']
                    best_lr = lr
            print(f'Лучшее значение accuracy для модели линейной регрессии с числом соседей {best_lr} CV: {best_accuracy.mean():.4f}, CV STD: {best_accuracy.std():.4f}')
        else:
            logistic_regression_model = LogisticRegression(C=constants.LOGISTIC_REGRESSION_MODEL['BEST_PARAMS']['lr'])

            pipeline = Pipeline([
                ('transformer', DataTransformer()),
                ('selector', ColumnSelector(columns_for_logistic_regression)),
                ('model', logistic_regression_model),
            ])

            logistic_regression_model_cv_results = cross_validate(
                pipeline, X_train, y_train,
                cv=cv,
                scoring=['accuracy']
            )
            print(f'Accuracy для модели линейной регрессии модели CV: {logistic_regression_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {logistic_regression_model_cv_results['test_accuracy'].std():.4f}')

            # Визуализация важности признаков
            best_model = LogisticRegression(C=constants.LOGISTIC_REGRESSION_MODEL['BEST_PARAMS']['lr'])
            transformer = DataTransformer()
            X_train = transformer.fit_transform(X_train)
            best_model.fit(X_train[columns_for_logistic_regression], y_train)
            coefs = best_model.coef_[0]
            feature_names = columns_for_logistic_regression
            sorted_idx = np.argsort(np.abs(coefs))[::-1]
            coefs_sorted = coefs[sorted_idx]
            features_sorted = [feature_names[i] for i in sorted_idx]
            plt.figure(figsize=(10, 6))
            colors = ['red' if c < 0 else 'green' for c in coefs_sorted]
            plt.barh(features_sorted, coefs_sorted, color=colors)
            plt.xlabel('Коэффициент (влияние на выживание)')
            plt.title(f'Важность признаков в логистической регрессии (C={constants.LOGISTIC_REGRESSION_MODEL['BEST_PARAMS']['lr']})')
            plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            plt.show()
    else:
        print('Модель линейной регрессии не обучена. Флаг LOGISTIC_REGRESSION_MODEL["ENABLED"]')
