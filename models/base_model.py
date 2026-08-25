from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

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