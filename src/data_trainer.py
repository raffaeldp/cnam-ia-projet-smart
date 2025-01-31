import os

import yaml
from ultralytics import YOLO


class DataTrainer:

    def train(self):
        print("Starting data training")
        # Load the model
        model = YOLO("yolo11n.pt")

        # use apple silicon gpu
        model.to(device="cuda:0")

        # Tune hyperparameters for 30 epochs
        if not (os.path.exists("./runs")):
            model.tune(
                data=os.path.abspath("./dataset/config.yaml"),
                epochs=30,
                iterations=1,
                optimizer="AdamW",
                plots=False,
                save=False,
                val=False,
            )

        # Charger les hyperparamètres depuis le fichier YAML
        with open("./runs/detect/tune/best_hyperparameters.yaml", "r") as file:
            best_hyperparameters = yaml.safe_load(file)

        # Train using config.yaml and hyperparameters
        # Configurer et lancer l'entraînement
        model.train(
            data=os.path.abspath(
                "./dataset/config.yaml"
            ),  # Chemin vers votre fichier config.yaml
            **best_hyperparameters,  # Appliquez les hyper-paramètres
            project="./results",
        )