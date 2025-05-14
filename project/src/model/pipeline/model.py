from model.pipeline.preparation import prepare_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle as pk
from config import model_settings
from loguru import logger
import pandas as pd


def build_model() -> None:
    logger.info("starting up model building pipeline")
    df = prepare_data()

    feature_names = [
        "area",
        "constraction_year",
        "bedrooms",
        "garden",
        "balcony_yes",
        "parking_yes",
        "furnished_yes",
        "garage_yes",
        "storage_yes",
    ]

    X, y = _get_x_y(df, col_x=feature_names)
    X_train, X_test, y_train, y_test = _split_train_test(X, y)
    rf = _train_model(X_train, y_train)
    score = _evaluate_model(rf, X_test, y_test)
    _save_model(rf)


def _get_x_y(
    dataframe: pd.DataFrame, col_x: list[str], col_y: str = "rent"
) -> tuple[pd.DataFrame, pd.Series]:

    logger.info(f"defining X and Y variables. \nX vars: {col_x}\ny vars: {col_y}")

    return dataframe[col_x], dataframe[col_y]


def _split_train_test(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("splitting data into train and test sets")
    return train_test_split(features, target, test_size=0.2)


def _train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:

    logger.info("training a model with hyperparameters")

    grid_space = {"n_estimators": [100, 200, 300], "max_depth": [3, 6, 9, 12]}

    logger.debug(f"grid space = {grid_space}")

    grid = GridSearchCV(
        RandomForestRegressor(), param_grid=grid_space, cv=5, scoring="r2"
    )

    model_grid = grid.fit(X_train, y_train)

    return model_grid.best_estimator_


def _evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    model_score = model.score(X_test, y_test)
    logger.info(f"evaluating model performance. SCORE={model_score}")
    return model_score


def _save_model(model):
    model_path = f"{model_settings.model_path}/{model_settings.model_name}"
    logger.info(f"saving model to a directory: {model_path}")
    with open(model_path, "wb") as model_file:
        pk.dump(model, model_file)
