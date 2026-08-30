import kagglehub

import constants


def download_data(output_dir='./dataset'):
    if constants.SHOULD_DOWNLOAD_DATA:
        kagglehub.competition_download('titanic', output_dir=output_dir, force_download=True)
    else:
        print('Загрузка данных не выполнена. Флаг SHOULD_DOWNLOAD_DATA False')
