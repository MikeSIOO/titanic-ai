import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import constants
from data.feature_engineering import DataTransformer
from final_nn.nn_utils import nn_fit_and_predict


def submit_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    if not constants.SUBMIT_VOTING_ENSEMBLE_RESULT:
        print('Сабмит не выполнен. Флаг SUBMIT_VOTING_ENSEMBLE_RESULT')
        return

    transformer = DataTransformer()

    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    # колонки для моделей
    columns_for_knn = ['Pclass_norm', 'Sex_male', 'Age_bin_norm', 'Fare_bin_norm', 'Fare_by_Pclass_bin_norm', 'Embarked_Q', 'Embarked_S', 'SibSp_norm', 'Parch_norm', 'Family_size_norm']
    columns_for_logistic_regression = ['Pclass_norm', 'Sex_male', 'Age_bin_norm', 'Fare_bin_norm', 'Fare_by_Pclass_bin_norm', 'Embarked_Q', 'Embarked_S', 'SibSp_norm', 'Parch_norm', 'Family_size_norm']
    columns_for_decision_tree = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
    columns_for_random_forest = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
    columns_for_cat_boost, cat_features_for_cat_boost = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Family_size', 'Sex', 'Embarked'], ['Sex', 'Embarked']
    columns_for_light_gbm = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
    columns_for_xg_boost = ['Pclass', 'Sex_male', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Family_size']
    columns_for_nn = ['Embarked_Q', 'Embarked_S', 'Sex_male', 'Pclass_norm', 'Age_norm', 'SibSp_norm', 'Parch_norm', 'Fare_norm', 'Family_size_norm']

    # создание моделей
    knn_model = KNeighborsClassifier(n_neighbors=constants.KNN_MODEL['BEST_PARAMS']['n_neighbors'])
    logistic_regression_model = LogisticRegression(C=constants.LOGISTIC_REGRESSION_MODEL['BEST_PARAMS']['lr'])
    decision_tree_model = DecisionTreeClassifier(
        criterion=constants.DECISION_TREE_MODEL['BEST_PARAMS']['criterion'],
        max_depth=constants.DECISION_TREE_MODEL['BEST_PARAMS']['max_depth'],
        min_samples_leaf=constants.DECISION_TREE_MODEL['BEST_PARAMS']['min_samples_leaf'],
        min_samples_split=constants.DECISION_TREE_MODEL['BEST_PARAMS']['min_samples_split']
    )
    random_forest_model = RandomForestClassifier(
        n_estimators=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['n_estimators'],
        criterion=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['criterion'],
        max_depth=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['max_depth'],
        min_samples_leaf=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['min_samples_leaf'],
        min_samples_split=constants.RANDOM_FOREST_MODEL['BEST_PARAMS']['min_samples_split'],
        random_state=constants.RANDOM_STATE
    )
    cat_boost_model = CatBoostClassifier(
        loss_function='Logloss',
        random_seed=constants.RANDOM_STATE,
        eval_metric='Accuracy',
        depth=constants.CAT_BOOST_MODEL['BEST_PARAMS']['depth'],
        l2_leaf_reg=constants.CAT_BOOST_MODEL['BEST_PARAMS']['l2_leaf_reg'],
        learning_rate=constants.CAT_BOOST_MODEL['BEST_PARAMS']['learning_rate'],
        iterations=constants.CAT_BOOST_MODEL['BEST_PARAMS']['iterations'],
        subsample=constants.CAT_BOOST_MODEL['BEST_PARAMS']['subsample'],
        grow_policy=constants.CAT_BOOST_MODEL['BEST_PARAMS']['grow_policy']
    )
    light_gbm_model = LGBMClassifier(
        random_state=constants.RANDOM_STATE,
        verbosity=-1,
        n_estimators=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['n_estimators'],
        learning_rate=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['learning_rate'],
        max_depth=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['max_depth'],
        num_leaves=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['num_leaves'],
        subsample=constants.LIGHT_GBM_MODEL['BEST_PARAMS']['subsample'],
    )
    xg_boost_model = XGBClassifier(
        random_state=constants.RANDOM_STATE,
        verbosity=0,
        max_depth=constants.XG_BOOST_MODEL['BEST_PARAMS']['max_depth'],
        learning_rate=constants.XG_BOOST_MODEL['BEST_PARAMS']['learning_rate'],
        n_estimators=constants.XG_BOOST_MODEL['BEST_PARAMS']['n_estimators'],
        subsample=constants.XG_BOOST_MODEL['BEST_PARAMS']['subsample']
    )

    # обучение моделей
    knn_model.fit(X_train_transformed[columns_for_knn], y_train)
    logistic_regression_model.fit(X_train_transformed[columns_for_logistic_regression], y_train)
    decision_tree_model.fit(X_train_transformed[columns_for_decision_tree], y_train)
    random_forest_model.fit(X_train_transformed[columns_for_random_forest], y_train)
    cat_boost_model.fit(X_train_transformed[columns_for_cat_boost], y_train, cat_features=cat_features_for_cat_boost)
    light_gbm_model.fit(X_train_transformed[columns_for_light_gbm], y_train)
    xg_boost_model.fit(X_train_transformed[columns_for_xg_boost], y_train)

    # предсказание
    knn_proba = knn_model.predict_proba(X_test_transformed[columns_for_knn])[:, 1]
    logistic_regression_proba = logistic_regression_model.predict_proba(X_test_transformed[columns_for_logistic_regression])[:, 1]
    decision_tree_proba = decision_tree_model.predict_proba(X_test_transformed[columns_for_decision_tree])[:, 1]
    random_forest_proba = random_forest_model.predict_proba(X_test_transformed[columns_for_random_forest])[:, 1]
    cat_boost_proba = cat_boost_model.predict_proba(X_test_transformed[columns_for_cat_boost])[:, 1]
    light_gbm_proba = light_gbm_model.predict_proba(X_test_transformed[columns_for_light_gbm])[:, 1]
    xg_boost_proba = xg_boost_model.predict_proba(X_test_transformed[columns_for_xg_boost])[:, 1]
    nn_proba = nn_fit_and_predict(
        X_train_transformed, y_train, X_test_transformed,
        columns_for_nn=columns_for_nn,
        hidden_sizes=constants.FINAL_MODEL['BEST_PARAMS']['hidden_sizes'],
        batchnorm=constants.FINAL_MODEL['BEST_PARAMS']['batchnorm'],
        dropout=constants.FINAL_MODEL['BEST_PARAMS']['dropout'],
        activation=constants.FINAL_MODEL['BEST_PARAMS']['activation'],
        optimizer_name=constants.FINAL_MODEL['BEST_PARAMS']['optimizer_name'],
        lr=constants.FINAL_MODEL['BEST_PARAMS']['lr'],
        weight_decay=constants.FINAL_MODEL['BEST_PARAMS']['weight_decay'],
        scheduler_name=constants.FINAL_MODEL['BEST_PARAMS']['scheduler_name'],
        batch_size=constants.FINAL_MODEL['BEST_PARAMS']['batch_size'],
        num_epochs=constants.FINAL_MODEL['BEST_PARAMS']['num_epochs'],
    )

    knn_pred = (knn_proba > 0.5).astype(int)
    logistic_regression_pred = (logistic_regression_proba > 0.5).astype(int)
    decision_tree_pred = (decision_tree_proba > 0.5).astype(int)
    random_forest_pred = (random_forest_proba > 0.5).astype(int)
    cat_boost_pred = (cat_boost_proba > 0.5).astype(int)
    light_gbm_pred = (light_gbm_proba > 0.5).astype(int)
    xg_boost_pred = (xg_boost_proba > 0.5).astype(int)
    nn_pred = (nn_proba > 0.5).astype(int)

    knn_acc = 0.7923
    logistic_regression_acc = 0.7991
    decision_tree_acc = 0.8215
    random_forest_acc = 0.8395
    cat_boost_acc = 0.8137
    light_gbm_acc = 0.8384
    xg_boost_acc = 0.8406
    nn_acc = 0.8249

    weighted_vote = ((knn_pred * knn_acc + logistic_regression_pred * logistic_regression_acc + decision_tree_pred * decision_tree_acc + random_forest_pred * random_forest_acc +
                      cat_boost_pred * cat_boost_acc + light_gbm_pred * light_gbm_acc + xg_boost_pred * xg_boost_acc + nn_pred * nn_acc) /
                     (knn_acc + logistic_regression_acc + decision_tree_acc + random_forest_acc + cat_boost_acc + light_gbm_acc + xg_boost_acc + nn_acc))
    predictions = (weighted_vote > 0.5).astype(int)

    output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': predictions})
    output.to_csv('./submission/voting_ensemble_submission.csv', index=False)