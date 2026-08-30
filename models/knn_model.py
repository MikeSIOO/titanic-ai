import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.neighbors import KNeighborsClassifier

import constants


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.KNN_MODEL['ENABLED']:
        columns_for_knn = ['Pclass_norm', 'Sex_male', 'Age_bin_norm', 'Fare_bin_norm', 'Fare_by_Pclass_bin_norm', 'Embarked_Q', 'Embarked_S', 'SibSp_norm', 'Parch_norm', 'Family_size_norm']
        if constants.KNN_MODEL['FIND_BEST_PARAMS']:
            possible_n_neighbors = [1, 3, 5, 7, 9, 11, 15, 21, 31]
            best_accuracy = None
            best_n_neighbors = None
            for n_neighbors in possible_n_neighbors:
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn_model_cv_results = cross_validate(
                    knn_model, X_train[columns_for_knn], y_train,
                    cv=KFold(constants.N_SPLITS),
                    scoring=['accuracy']
                )
                print(f'Accuracy для KNN модели с числом соседей {n_neighbors} CV: {knn_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {knn_model_cv_results['test_accuracy'].std():.4f}')
                if best_accuracy is None or best_accuracy.mean() < knn_model_cv_results['test_accuracy'].mean():
                    best_accuracy = knn_model_cv_results['test_accuracy']
                    best_n_neighbors = n_neighbors
            print(f'Лучшее значение accuracy для KNN модели с числом соседей {best_n_neighbors} CV: {best_accuracy.mean():.4f}, CV STD: {best_accuracy.std():.4f}')
        else:
            knn_model = KNeighborsClassifier(n_neighbors=constants.KNN_MODEL['BEST_PARAMS']['n_neighbors'])
            knn_model_cv_results = cross_validate(
                knn_model, X_train[columns_for_knn], y_train,
                cv=KFold(constants.N_SPLITS),
                scoring=['accuracy']
            )
            print(f'Accuracy для KNN модели CV: {knn_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {knn_model_cv_results['test_accuracy'].std():.4f}')
    else:
        print('KNN модель не обучена. Флаг KNN_MODEL["ENABLED"]')
