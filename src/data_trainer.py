import os
from pathlib import Path

from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from config import settings
from src.data_preprocessor import DataPreprocessor

class PicselliaCallback:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def on_train_epoch_end(self, trainer: DetectionTrainer):
        print(str(trainer.metrics))
        self.experiment.log(str(trainer.epoch), "test", LogType.LINE)


class DataTrainer:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def train(self):
        print("Starting data training")
        # Load the model
        model = YOLO("yolo11s.pt")

        callback = PicselliaCallback(self.experiment)

        model.add_callback("on_train_epoch_end",callback.on_train_epoch_end)

        # Train the model using the 'data.yaml' datasets for 3 epochs
        config_path = Path(DataPreprocessor.config_path).resolve()

        results = model.train(
            data=config_path,
            device=settings.get("training_device"),
            imgsz=640,
            epochs=50,
            batch=16,
            close_mosaic=False,
            optimizer="adamW",
            seed=42,
            lr0=0.001,
        )

        # Evaluate the model's performance on the val set
        results = model.val()