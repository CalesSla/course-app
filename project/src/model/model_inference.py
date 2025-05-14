"""
Short desc

Detailed desc
"""

from pathlib import Path
import pickle as pk

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelInferenceService:
    """
    Brief overview

    Detailed overview

    Attributes:

    Methods:
    """

    def __init__(self) -> None:
        self.model = None
        self.model_path = model_settings.model_path
        self.model_name = model_settings.model_name

    def load_model(self) -> None:
        logger.info(
            f"checking the existence of model config file at {self.model_path}/{self.model_name}"
        )
        model_path = Path(f"{self.model_path}/{self.model_name}")

        if not model_path.exists():
            raise FileNotFoundError("Model file does not exist!")        

        logger.info(
            f"model {self.model_name} exists! -> "
            f"loading model configuration file"
        )

        with open(model_path, "rb") as file:
            self.model = pk.load(file)

    def predict(self, input_parameters: list) -> list:
        logger.info(f"making prediction!")
        return self.model.predict([input_parameters])
