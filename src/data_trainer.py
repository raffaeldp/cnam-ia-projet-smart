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
        for key, value in trainer.metrics.items():
            self.experiment.log(f"{trainer.epoch}_{key}", value, LogType.VALUE)


class DataTrainer:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.model = YOLO("yolo11s.pt")

        self.callback = PicselliaCallback(self.experiment)
        self.model.add_callback("on_train_epoch_end", self.callback.on_train_epoch_end)


    def train(self):
        print("Starting data training")

        # Train the model using the 'data.yaml' datasets for 3 epochs
        config_path = Path(DataPreprocessor.config_path).resolve()

        results = self.model.train(
            data=config_path,
            device=settings.get("training_device"),
            imgsz=640,
            epochs=2,
            batch=16,
            close_mosaic=False,
            optimizer="adamW",
            seed=42,
            lr0=0.001,
        )