from pathlib import Path

from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from src.data_preprocessor import DataPreprocessor
from src.utils import get_model_device


class PicselliaCallback:
    """
    Callback class to log training metrics to Picsellia.

    Attributes:
        experiment (Experiment): The Picsellia experiment instance.
    """

    def __init__(self, experiment: Experiment) -> None:
        """
        Initializes the PicselliaCallback with the given experiment.

        Args:
            experiment (Experiment): The Picsellia experiment instance.
        """
        self.experiment = experiment

    def on_train_epoch_end(self, trainer: DetectionTrainer) -> None:
        """
        Logs the training metrics to Picsellia at the end of each training epoch.

        Args:
            trainer (DetectionTrainer): The YOLO detection trainer instance.
        """
        for key, value in trainer.metrics.items():
            try:
                self.experiment.log(f"{trainer.epoch}_{key}", value, LogType.VALUE)
            except Exception as e:
                print(e)


class DataTrainer:
    """
    Class to handle the training of a YOLO model with Picsellia integration.

    Attributes:
        experiment (Experiment): The Picsellia experiment instance.
        model (YOLO): The YOLO model instance.
        callback (PicselliaCallback): The callback instance for logging metrics.
    """

    def __init__(self, experiment: Experiment) -> None:
        """
        Initializes the DataTrainer with the given experiment and sets up the YOLO model.

        Args:
            experiment (Experiment): The Picsellia experiment instance.
        """
        self.experiment = experiment
        self.model = YOLO("yolo11s.pt")

        self.callback = PicselliaCallback(self.experiment)
        self.model.add_callback("on_train_epoch_end", self.callback.on_train_epoch_end)

    def train(self) -> None:
        """
        Starts the training process for the YOLO model.
        """
        print("Starting data training")

        # Train the model using the 'data.yaml' datasets for 3 epochs
        config_path = Path(DataPreprocessor.config_path).resolve()

        self.model.train(
            data=config_path,
            device=get_model_device(),
            imgsz=640,
            epochs=400,
            batch=16,
            close_mosaic=False,
            optimizer="adamW",
            seed=42,
            lr0=0.001,
        )
