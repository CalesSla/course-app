"""
Short desc

Detailed desc
"""

from pathlib import Path
import pickle as pk

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelService:
    """
    Brief overview

    Detailed overview

    Attributes:

    Methods:
    """

    def __init__(self) -> None:
        self.model = None

    def load_model(self) -> None:
        logger.info(
            f"checking the existence of model config file at {model_settings.model_path}/{model_settings.model_name}"
        )
        model_path = Path(f"{model_settings.model_path}/{model_settings.model_name}")

        if not model_path.exists():
            logger.warning(
                f"model at {model_settings.model_path}/{model_settings.model_name} was not found -> "
                f"building {model_settings.model_name}"
            )
            build_model()

        logger.info(
            f"model {model_settings.model_name} exists! -> "
            f"loading model configuration file"
        )

        with open(model_path, "rb") as file:
            self.model = pk.load(file)

    def predict(self, input_parameters: list) -> list:
        logger.info(f"making prediction!")
        return self.model.predict([input_parameters])
