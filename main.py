import pandas as pd

import data.analyze_data
import data.get_data
import settings
from data import feature_engineering

if __name__ == '__main__':
    # Установка настроек пандаса (показ всех колонок таблицы)
    settings.set_pandas_settings()

    # Скачивание данных
    data.get_data.download_data()

    # Вывести различные метрики для анализа данных
    data.analyze_data.print_train_data()

    # Создаем и обучаем трансформер для предобработки данных
    data_transformer = feature_engineering.DataTransformer()
    train_data = pd.read_csv("./dataset/train.csv", index_col='PassengerId')
    data_transformer.fit()

    test_data = pd.read_csv("./dataset/test.csv", index_col='PassengerId')
    transformed_test_data = data_transformer.transform(test_data)
    print(transformed_test_data.head())
    print(transformed_test_data.info())
