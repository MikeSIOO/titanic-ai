import pandas as pd

import data.analyze_data
import data.get_data
import settings
from linear_models import base_model, knn_model, logistic_regression_model, decision_tree_model, random_forest_model, \
    cat_boost_model, light_gbm_model, xg_boost_model
from nn_models import two_layer_simple_model

if __name__ == '__main__':
    # Установка настроек пандаса (показ всех колонок таблицы)
    settings.set_pandas_settings()

    # Скачивание данных
    data.get_data.download_data()

    # Вывести различные метрики для анализа данных
    data.analyze_data.print_train_data()

    # Загрузка данных из файла
    train_data = pd.read_csv("./dataset/train.csv", index_col='PassengerId')
    # Разделение признаков и таргета
    X_train = train_data.drop('Survived', axis=1)
    y_train = train_data['Survived']

    # Базовая модель
    base_model.run_model(X_train, y_train)

    # Метод ближайших соседей
    knn_model.run_model(X_train, y_train)

    # Метод логистической регрессии
    logistic_regression_model.run_model(X_train, y_train)

    # Решающее дерево
    decision_tree_model.run_model(X_train, y_train)

    # Случайный лес
    random_forest_model.run_model(X_train, y_train)

    # Градиентный бустинг (CatBoost)
    cat_boost_model.run_model(X_train, y_train)

    # Градиентный бустинг (LightGBM)
    light_gbm_model.run_model(X_train, y_train)

    # Градиентный бустинг (XGBoost)
    xg_boost_model.run_model(X_train, y_train)

    # Нейросеть с двумя слоями
    two_layer_simple_model.run_model(X_train, y_train)
