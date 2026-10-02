import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_bins: int = 10,
                 one_hot_columns=None,
                 scale_columns=None,
                 drop_columns=None):
        self.n_bins = n_bins
        self.one_hot_columns = one_hot_columns
        self.scale_columns = scale_columns
        self.drop_columns = drop_columns

    def fit(self, X: pd.DataFrame, y=None) -> 'DataTransformer':
        self.one_hot_columns_ = self.one_hot_columns or ['Embarked', 'Sex']
        self.scale_columns_ = self.scale_columns or [
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            'Age_bin', 'Fare_bin', 'Fare_by_Pclass_bin', 'Family_size',
        ]
        self.drop_columns_ = self.drop_columns or ['Name', 'Ticket', 'Cabin']

        df = X.copy()

        # Средние для заполнения пропусков
        self.age_means_by_group_ = df.groupby(['Pclass', 'SibSp', 'Parch'])['Age'].mean()
        self.age_means_by_pclass_ = df.groupby('Pclass')['Age'].mean()
        self.age_global_mean_ = df['Age'].mean()

        self.fare_means_by_pclass_ = df.groupby('Pclass')['Fare'].mean()
        self.fare_global_mean_ = df['Fare'].mean()

        # Заполняем пропуски
        df = self.__fill_na(df)

        # Границы бинов
        self.__fit_bins(df)

        # Family_size
        df = self.__sum_family_members(df)

        # OneHotEncoder
        self.one_hot_encoder_ = OneHotEncoder(
            handle_unknown='ignore', drop='first', sparse_output=False
        )
        self.one_hot_encoder_.fit(df[self.one_hot_columns_])

        # Бины (для скалера)
        df = self.__apply_bins(df)

        # StandardScaler
        self.scaler_ = StandardScaler()
        self.scaler_.fit(df[self.scale_columns_])

        return self

    def transform(self, X):
        df = X.copy()
        df = self.__fill_na(df)
        df = self.__apply_bins(df)
        df = self.__sum_family_members(df)

        one_hot_df = pd.DataFrame(
            self.one_hot_encoder_.transform(df[self.one_hot_columns_]),
            columns=self.one_hot_encoder_.get_feature_names_out(self.one_hot_columns_),
            index=df.index,
        ).astype(int)

        scaled_df = pd.DataFrame(
            self.scaler_.transform(df[self.scale_columns_]),
            columns=[c + '_norm' for c in self.scale_columns_],
            index=df.index,
        )

        df = pd.concat([df, one_hot_df, scaled_df], axis=1)
        df = self.__drop_columns(df)
        return df

    def __fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def fill_age(row):
            if pd.notna(row['Age']):
                return row['Age']
            key = (row['Pclass'], row['SibSp'], row['Parch'])
            if key in self.age_means_by_group_.index:
                mean_val = self.age_means_by_group_.loc[key]
                if pd.notna(mean_val):
                    return float(mean_val)
            if row['Pclass'] in self.age_means_by_pclass_.index:
                mean_val = self.age_means_by_pclass_.loc[row['Pclass']]
                if pd.notna(mean_val):
                    return float(mean_val)
            return self.age_global_mean_
        df['Age'] = df.apply(fill_age, axis=1)

        def fill_fare(row):
            if pd.notna(row['Fare']):
                return row['Fare']
            if row['Pclass'] in self.fare_means_by_pclass_.index:
                mean_val = self.fare_means_by_pclass_.loc[row['Pclass']]
                if pd.notna(mean_val):
                    return float(mean_val)
            return self.fare_global_mean_
        df['Fare'] = df.apply(fill_fare, axis=1)

        df['Embarked'] = df['Embarked'].fillna('S')
        df['Sex'] = df['Sex'].fillna('female')
        df['SibSp'] = df['SibSp'].fillna(0)
        df['Parch'] = df['Parch'].fillna(0)

        return df

    def __fit_bins(self, df: pd.DataFrame) -> None:
        # Age
        self.age_bins_ = np.linspace(
            df['Age'].min() - 1e-6, df['Age'].max() + 1e-6, self.n_bins + 1
        )
        # Fare
        self.fare_bins_ = np.linspace(
            df['Fare'].min() - 1e-6, df['Fare'].max() + 1e-6, self.n_bins + 1
        )
        # Fare по Pclass
        self.fare_by_pclass_bins_ = {}
        for pclass in df['Pclass'].unique():
            fares = df.loc[df['Pclass'] == pclass, 'Fare']
            self.fare_by_pclass_bins_[pclass] = np.linspace(
                fares.min() - 1e-6, fares.max() + 1e-6, self.n_bins + 1
            )

    def __apply_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Age
        df['Age_bin'] = pd.cut(
            df['Age'].clip(self.age_bins_[0], self.age_bins_[-1]),
            bins=self.age_bins_, labels=False, include_lowest=True,
        ).astype(int)

        # Fare
        df['Fare_bin'] = pd.cut(
            df['Fare'].clip(self.fare_bins_[0], self.fare_bins_[-1]),
            bins=self.fare_bins_, labels=False, include_lowest=True,
        ).astype(int)

        # Fare_by_Pclass_bin
        df['Fare_by_Pclass_bin'] = 0
        for pclass, bins in self.fare_by_pclass_bins_.items():
            mask = df['Pclass'] == pclass
            if mask.any():
                fares_clipped = df.loc[mask, 'Fare'].clip(bins[0], bins[-1])
                df.loc[mask, 'Fare_by_Pclass_bin'] = pd.cut(
                    fares_clipped, bins=bins, labels=False, include_lowest=True,
                ).astype(int)

        return df

    @staticmethod
    def __sum_family_members(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Family_size'] = df['SibSp'] + df['Parch']
        return df

    def __drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols_to_drop = [c for c in self.drop_columns_ if c in df.columns]
        df.drop(cols_to_drop, axis=1, inplace=True)
        return df
