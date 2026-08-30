import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate, KFold

import constants


def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    if constants.BASE_MODEL['ENABLED']:
        base_model = BaseModel()
        base_model_cv_results = cross_validate(
            base_model, X_train, y_train,
            cv=KFold(constants.N_SPLITS),
            scoring=['accuracy']
        )
        print(f'Accuracy для базовой модели CV: {base_model_cv_results['test_accuracy'].mean():.4f}, CV STD: {base_model_cv_results['test_accuracy'].std():.4f}')
    else:
        print('Базовая модель не обучена. Флаг BASE_MODEL["ENABLED"]')


class BaseModel(BaseEstimator, ClassifierMixin):
    def __init__(self, best_prediction=0):
        self.best_prediction = best_prediction

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        self.best_prediction = 1 if y.mean() > 0.5 else 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.best_prediction, dtype=int)

    # Заглушки так как пришлось наследоваться от классов sklearn для удобной кросс валидации
    def get_params(self, deep=True):
        return {'best_prediction': self.best_prediction}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
