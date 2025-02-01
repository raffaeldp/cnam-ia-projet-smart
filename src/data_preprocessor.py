import os
import random
from glob import glob

import yaml


class DataPreprocessor:
    processed_folder_path = "./datasets/processed"
    images_path = f"{processed_folder_path}/images"
    labels_path = f"{processed_folder_path}/labels"

    images_train_path = f"{images_path}/train"
    images_validation_path = f"{images_path}/val"
    images_test_path = f"{images_path}/test"

    labels_train_path = f"{labels_path}/train"
    labels_validation_path = f"{labels_path}/val"
    labels_test_path = f"{labels_path}/test"

    random.seed(42)

    split_ratio = {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2
    }

    def pre_process(self):
        print("Starting data pre-processing")

        if os.path.exists(self.processed_folder_path):
            print("Data already pre-processed")
            return

        # Create processed directory if not exists
        self.ensure_folders_exists()

        # Get all images and labels
        images = glob("./datasets/images/*")
        labels = glob("./datasets/annotations/*.txt")

        # Pair images and labels
        pair_images_labels = list(zip(images, labels))
        random.shuffle(pair_images_labels)

        # Split data
        train_index = int(len(pair_images_labels) * self.split_ratio["train"])
        validation_index = int(len(pair_images_labels) * (self.split_ratio["train"] + self.split_ratio["val"]))

        train_data = pair_images_labels[:train_index]
        validation_data = pair_images_labels[train_index:validation_index]
        test_data = pair_images_labels[validation_index:]

        # Copy images and labels to the right folder
        self.copy_images_and_labels(train_data, self.images_train_path, self.labels_train_path)
        self.copy_images_and_labels(validation_data, self.images_validation_path, self.labels_validation_path)
        self.copy_images_and_labels(test_data, self.images_test_path, self.labels_test_path)

        # Generate data.yaml file
        self.generate_yaml_file("./datasets/data.yaml")
        print("Data pre-processing finished")

    def ensure_folders_exists(self):
        os.makedirs(self.processed_folder_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)

        os.makedirs(self.images_train_path, exist_ok=True)
        os.makedirs(self.images_validation_path, exist_ok=True)
        os.makedirs(self.images_test_path, exist_ok=True)

        os.makedirs(self.labels_train_path, exist_ok=True)
        os.makedirs(self.labels_validation_path, exist_ok=True)
        os.makedirs(self.labels_test_path, exist_ok=True)

    def copy_images_and_labels(self, data, images_path, labels_path):
        for image, label in data:
            image_name = os.path.basename(image)
            label_name = os.path.basename(label)

            image_dst = f"{images_path}/{image_name}"
            label_dst = f"{labels_path}/{label_name}"

            os.system(f"cp {image} {image_dst}")
            os.system(f"cp {label} {label_dst}")

    def generate_yaml_file(self, result_path):
        data = {
            "train": f"{self.images_train_path}",
            "val": f"{self.images_validation_path}",
            "test": f"{self.images_test_path}",
            "nc": 10,  # Classes count
            "names": [
                "mikado",
                "kinder_pingui",
                "kinder_country",
                "kinder_tronky",
                "tic_tac",
                "sucette",
                "capsule",
                "pepito",
                "bouteille_plastique",
                "canette",
            ],
        }
        with open(result_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
        print(f"Yaml file generated : {result_path}")

