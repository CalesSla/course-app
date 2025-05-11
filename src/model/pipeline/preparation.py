import re
import warnings

import pandas as pd
from loguru import logger

from model.pipeline.collection import load_data_from_db

warnings.filterwarnings("ignore")


def prepare_data():
    logger.info("starting up preprocessing pipeline")
    data = load_data_from_db()
    data_encoded = _encode_cat_cols(data)
    df = _parse_garden_col(data_encoded)
    return df


def _encode_cat_cols(dataframe: pd.DataFrame) -> pd.DataFrame:
    cols = ["balcony", "parking", "furnished", "garage", "storage"]
    logger.info(f"encoding categorical columns {cols}")
    return pd.get_dummies(dataframe, columns=cols, drop_first=True, dtype=int)


def _parse_garden_col(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("parsing garden column")
    dataframe["garden"] = dataframe["garden"].apply(
        lambda x: 0 if x == "Not present" else int(re.findall(r"\d+", x)[0])
    )
    return dataframe
