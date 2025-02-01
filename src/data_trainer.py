import os
from pathlib import Path

import yaml
from torch.xpu import device
from ultralytics import YOLO


class DataTrainer:

    def train(self):
        print("Starting data training")
        # Load the model
        model = YOLO("yolo11n.pt")


        # Train the model using the 'data.yaml' datasets for 3 epochs

        config_path = Path("./datasets/data.yaml").resolve()
        results = model.train(data=config_path, epochs=3, device="mps")

        # Evaluate the model's performance on the val set
        results = model.val()

        # # use apple silicon gpu
        # model.to(device="cuda:0")
        #
        # # Tune hyperparameters for 30 epochs
        # if not (os.path.exists("./runs")):
        #     model.tune(
        #         data=os.path.abspath("./datasets/data.yaml"),
        #         epochs=30,
        #         iterations=1,
        #         optimizer="AdamW",
        #         plots=False,
        #         save=False,
        #         val=False,
        #     )
        #
        # # Charger les hyperparamètres depuis le fichier YAML
        # with open("./runs/detect/tune/best_hyperparameters.yaml", "r") as file:
        #     best_hyperparameters = yaml.safe_load(file)
        #
        # # Train using data.yaml and hyperparameters
        # # Configurer et lancer l'entraînement
        # model.train(
        #     data=os.path.abspath(
        #         "./datasets/data.yaml"
        #     ),  # Chemin vers votre fichier data.yaml
        #     **best_hyperparameters,  # Appliquez les hyper-paramètres
        #     project="./results",
        # )