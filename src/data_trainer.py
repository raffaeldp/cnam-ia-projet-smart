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
        model = YOLO("yolo11n.pt")


        # Train the model using the 'data.yaml' datasets for 3 epochs

        config_path = Path(DataPreprocessor.config_path).resolve()
        results = model.train(data=config_path, epochs=3, device="mps")

        # Evaluate the model's performance on the val set
        results = model.val()