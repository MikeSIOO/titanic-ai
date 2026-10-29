from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

import constants
from data.feature_engineering import DataTransformer


def submit_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    if not constants.SUBMIT_LIGHT_GBM_RESULT:
        print('Сабмит не выполнен. Флаг SUBMIT_LIGHT_GBM_RESULT')
        return

    columns = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
    model = LGBMClassifier(
        random_state=constants.RANDOM_STATE,
        verbosity=-1,
        n_estimators=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['n_estimators'],
        learning_rate=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['learning_rate'],
        max_depth=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['max_depth'],
        num_leaves=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['num_leaves'],
        subsample=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['subsample'],
    )
    transformer = DataTransformer()

    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    model.fit(X_train_transformed[columns], y_train)
    predictions = model.predict(X_test_transformed[columns])

    output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': predictions})
    output_path = Path('./submission/light_gbm_submission.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    output.to_csv('./submission/light_gbm_submission.csv', index=False)

