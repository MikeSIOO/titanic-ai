import pandas as pd

import data.analyze_data
import data.get_data
import settings
from data import feature_engineering
from models import base_model, knn_model, logistic_regression

if __name__ == '__main__':
    # Установка настроек пандаса (показ всех колонок таблицы)
    settings.set_pandas_settings()

    # Скачивание данных
    data.get_data.download_data()

    # Вывести различные метрики для анализа данных
    data.analyze_data.print_train_data()

    # Загрузка данных из файла
    train_data = pd.read_csv("./dataset/train.csv", index_col='PassengerId')
    test_data = pd.read_csv("./dataset/test.csv", index_col='PassengerId')
    # Разделение признаков и таргета
    X_train = train_data.drop('Survived', axis=1)
    y_train = train_data['Survived']
    X_test = test_data

    # Создаем и обучаем трансформер для предобработки данных
    data_transformer = feature_engineering.DataTransformer()
    X_train = data_transformer.fit_transform(X_train)

    # Предобработка данных
    X_test = data_transformer.transform(X_test)

    # Базовая модель
    base_model.run_model(X_train, y_train)

    # Метод ближайших соседей
    knn_model.run_model(X_train, y_train)

    # Метод логистической регрессии
    logistic_regression.run_model(X_train, y_train)
