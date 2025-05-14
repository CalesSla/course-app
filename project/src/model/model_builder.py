"""
Short desc

Detailed desc
"""

from pathlib import Path
import pickle as pk

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelBuilderService:
    """
    Brief overview

    Detailed overview

    Attributes:

    Methods:
    """

    def __init__(self) -> None:
        self.model_path = model_settings.model_path
        self.model_name = model_settings.model_name

    def train_model(self) -> None:
        logger.info(
            f"building the model of model config file at {model_settings.model_path}/{model_settings.model_name}"
        )
        build_model()