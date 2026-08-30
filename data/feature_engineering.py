import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataTransformer:

    def __init__(self):
        self.one_hot_encoder = None
        self.age_bins = None
        self.fare_by_pclass_bins = {}

    def fit_transform(self, train_data: pd.DataFrame) -> pd.DataFrame:
        self.original_train_data = train_data
        self.train_data = train_data

        self.__fill_na(self.train_data)
        self.__set_bins_for_cut()
        self.__sum_family_members(self.train_data)
        self.__set_one_hot_encoder()
        self.__set_norm_const()
        self.__drop_columns(self.train_data)

        return self.train_data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_to_transform = df.copy()

        self.__fill_na(df_to_transform)
        self.__cut_by_bins(df_to_transform)
        self.__sum_family_members(df_to_transform)

        one_hot_df = pd.DataFrame(
            self.one_hot_encoder.transform(df_to_transform[['Embarked', 'Sex']]),
            columns=self.one_hot_encoder.get_feature_names_out(['Embarked', 'Sex']),
            index=df_to_transform.index
        ).astype(int)

        scaled_df = pd.DataFrame(
            self.scaler.fit_transform(df_to_transform[['Pclass', 'SibSp', 'Parch', 'Age_bin', 'Fare_bin', 'Fare_by_Pclass_bin', 'Family_size']]),
            columns=['Pclass_norm', 'SibSp_norm', 'Parch_norm', 'Age_bin_norm', 'Fare_bin_norm', 'Fare_by_Pclass_bin_norm', 'Family_size_norm'],
            index=df_to_transform.index
        )

        df_to_transform = pd.concat([df_to_transform, one_hot_df, scaled_df], axis=1)

        self.__drop_columns(df_to_transform)

        return df_to_transform

    def __set_bins_for_cut(self) -> None:
        # Разбиваем возраст на бины в трейне и сохраняем границы, чтобы по ним же разбить на тесте
        self.age_bins = np.linspace(self.train_data['Age'].min() - 1e-6, self.train_data['Age'].max() + 1e-6, 11)
        self.train_data['Age_bin'] = pd.cut(self.train_data['Age'], bins=self.age_bins, labels=False, include_lowest=True).astype(int)
        # Разбиваем стоимость билета на бины в трейне и сохраняем границы, чтобы по ним же разбить на тесте
        self.fare_bins = np.linspace(self.train_data['Fare'].min() - 1e-6, self.train_data['Fare'].max() + 1e-6, 11)
        self.train_data['Fare_bin'] = pd.cut(self.train_data['Fare'], bins=self.fare_bins, labels=False, include_lowest=True)
        # Разбиваем стоимость билета для каждого класса отдельно на бины в трейне и сохраняем границы, чтобы по ним же разбить на тесте
        for pclass in self.train_data['Pclass'].unique():
            fares = self.train_data[self.train_data['Pclass'] == pclass]['Fare']
            self.fare_by_pclass_bins[pclass] = np.linspace(fares.min() - 1e-6, fares.max() + 1e-6, 11)
            self.train_data.loc[self.train_data['Pclass'] == pclass, 'Fare_by_Pclass_bin'] = pd.cut(fares, bins=self.fare_by_pclass_bins[pclass], labels=False, include_lowest=True).astype('Int64')
        return

    def __cut_by_bins(self, df: pd.DataFrame) -> None:
        # Разбиваем возраст в тесте на бины по сохраненным границам
        df['Age_bin'] = pd.cut(df['Age'].clip(self.age_bins[0], self.age_bins[-1]), bins=self.age_bins, labels=False, include_lowest=True)
        # Разбиваем стоимость билета в тесте на бины по сохраненным границам
        df['Fare_bin'] = pd.cut(df['Fare'].clip(self.fare_bins[0], self.fare_bins[-1]), bins=self.fare_bins, labels=False, include_lowest=True)
        # Разбиваем стоимость билета для каждого класса отдельно в тесте на бины по сохраненным границам
        for pclass in df['Pclass'].unique():
            fares = df[df['Pclass'] == pclass]['Fare']
            df.loc[df['Pclass'] == pclass, 'Fare_by_Pclass_bin'] = pd.cut(fares.clip(self.fare_by_pclass_bins[pclass][0], self.fare_by_pclass_bins[pclass][-1]), bins=self.fare_by_pclass_bins[pclass], labels=False, include_lowest=True).astype('Int64')
        return

    def __set_one_hot_encoder(self) -> None:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

        one_hot_df = pd.DataFrame(
            one_hot_encoder.fit_transform(self.train_data[['Embarked', 'Sex']]),
            columns=one_hot_encoder.get_feature_names_out(['Embarked', 'Sex']),
            index=self.train_data.index
        ).astype(int)
        self.train_data = pd.concat([self.train_data, one_hot_df], axis=1)
        self.one_hot_encoder = one_hot_encoder
        return

    def __set_norm_const(self) -> None:
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(
            scaler.fit_transform(self.train_data[['Pclass', 'SibSp', 'Parch', 'Age_bin', 'Fare_bin', 'Fare_by_Pclass_bin', 'Family_size']]),
            columns=['Pclass_norm', 'SibSp_norm', 'Parch_norm', 'Age_bin_norm', 'Fare_bin_norm', 'Fare_by_Pclass_bin_norm', 'Family_size_norm'],
            index=self.train_data.index
        )
        self.train_data = pd.concat([self.train_data, scaled_df], axis=1)
        self.scaler = scaler
        return

    @staticmethod
    def __sum_family_members(df: pd.DataFrame) -> None:
        df['Family_size'] = df['SibSp'] + df['Parch']
        return

    @staticmethod
    def __fill_na(df: pd.DataFrame) -> None:
        # заполнение пропусков на основе среднего, сгруппированного по 3 признакам, показавшим наилучшую корреляцию
        df['Age'] = df.groupby(['Pclass', 'SibSp', 'Parch'])['Age'].transform(
            lambda x: x.fillna(x.mean())
        )
        # заполнение оставшихся пропусков (на всякий случай) на основе толко класса билета
        df['Age'] = df.groupby(['Pclass'])['Age'].transform(
            lambda x: x.fillna(x.mean())
        )
        # заполнение порта константой, так как мало пропусков
        df.fillna({'Embarked': 'S'}, inplace=True)

        # в трейне не было этих пропусков, но заполним на всякий случай для теста
        df.fillna({'Sex': 'female'}, inplace=True)
        df.fillna({'SibSp': 0}, inplace=True)
        df.fillna({'Parch': 0}, inplace=True)
        df['Fare'] = df.groupby(['Pclass'])['Fare'].transform(
            lambda x: x.fillna(x.mean())
        )
        return

    @staticmethod
    def __drop_columns(df: pd.DataFrame) -> None:
        # к сожалению, пока не придумал, что можно сделать с именем
        df.drop('Name', axis=1, inplace=True)

        # Для числовых признаков выполнена нормализация, исходные колонки можно дропнуть
        df.drop(['SibSp', 'Parch', 'Family_size'], axis=1, inplace=True)

        # к сожалению, пока не придумал, что можно сделать с билетом
        df.drop('Ticket', axis=1, inplace=True)

        # слишком много пропусков в данных, точно удаляем
        df.drop('Cabin', axis=1, inplace=True)

        # Категориальные признаки (Sex, Embarked) заменены на one hot пердставление, но не все деревья умеют с ним работать, так что оставим
        # непрерывные признаки (Age, Fare) тоже удалять не буду, предполажу, что деревья лучше с ними работают
        # SibSp и Parch ценны как вместе (Family_size) так и пооттдельности хочется посмотреть на их влияние. Особенно учитывая, что у полей была хорошая корреляция с выживаемостью
        # Нормаированные признаки не заменят порядковые (Pclass, Age_bin, Fare_bin, Fare_by_Pclass_bin) потому что мне инетерснопроверить модели в двух вариантах
        return
