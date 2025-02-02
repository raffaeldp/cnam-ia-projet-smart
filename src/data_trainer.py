import os
from pathlib import Path

import yaml
from torch.xpu import device
from ultralytics import YOLO

from src.data_preprocessor import DataPreprocessor


class DataTrainer:

    def train(self):
        print("Starting data training")
        # Load the model
        model = YOLO("yolo11s.pt")


        # Train the model using the 'data.yaml' datasets for 3 epochs

        config_path = Path(DataPreprocessor.config_path).resolve()

        results = model.train(
            data=config_path,
            device="mps",
            imgsz=640,
            epochs=50,
            batch=16,
            close_mosaic=False,
            optimizer="adamW",
            seed=42,
            lr0=0.001,
            lrf=0.001,
        )

        # Evaluate the model's performance on the val set
        results = model.val()