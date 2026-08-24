import pandas as pd
import numpy as np

import data.get_data
import data.analyze_data

if __name__ == '__main__':
    # Скачивание данных
    data.get_data.download_data()



    # Вывести различные метрики для анализа данных
    data.analyze_data.print_train_data()
