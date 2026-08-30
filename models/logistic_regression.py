import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, KFold

import constants


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.LOGISTIC_REGRESSION_MODEL['ENABLED']:
        columns_for_logistic_regression = ['Pclass_norm', 'Sex_male', 'Age_bin_norm', 'Fare_bin_norm', 'Fare_by_Pclass_bin_norm', 'Embarked_Q', 'Embarked_S', 'SibSp_norm', 'Parch_norm', 'Family_size_norm']
        if constants.LOGISTIC_REGRESSION_MODEL['FIND_BEST_PARAMS']:
            possible_lr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]
            best_accuracy = None
            best_lr = None
            for lr in possible_lr:
                logistic_regression_model = LogisticRegression(C=lr)
                logistic_regression_model_cv_results = cross_validate(
                    logistic_regression_model, X_train[columns_for_logistic_regression], y_train,
                    cv=KFold(constants.N_SPLITS),
                    scoring=['accuracy']
                )
                print(f'Accuracy для модели линейной регрессии с числом соседей {lr} CV: {logistic_regression_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {logistic_regression_model_cv_results['test_accuracy'].std():.4f}')
                if best_accuracy is None or best_accuracy.mean() < logistic_regression_model_cv_results['test_accuracy'].mean():
                    best_accuracy = logistic_regression_model_cv_results['test_accuracy']
                    best_lr = lr
            print(f'Лучшее значение accuracy для модели линейной регрессии с числом соседей {best_lr} CV: {best_accuracy.mean():.4f}, CV STD: {best_accuracy.std():.4f}')
        else:
            logistic_regression_model = LogisticRegression(C=constants.LOGISTIC_REGRESSION_MODEL['BEST_PARAMS']['lr'])
            logistic_regression_model_cv_results = cross_validate(
                logistic_regression_model, X_train[columns_for_logistic_regression], y_train,
                cv=KFold(constants.N_SPLITS),
                scoring=['accuracy']
            )
            print(f'Accuracy для модели линейной регрессии модели CV: {logistic_regression_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {logistic_regression_model_cv_results['test_accuracy'].std():.4f}')
    else:
        print('Модель линейной регрессии не обучена. Флаг LOGISTIC_REGRESSION_MODEL["ENABLED"]')
