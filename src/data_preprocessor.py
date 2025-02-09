import os
import random
from glob import glob
import shutil

import yaml

from src.data_downloader import DataDownloader


class DataPreprocessor:
    """
    Class to handle the preprocessing of data, including splitting datasets and generating configuration files.

    Attributes:
        dataset_path (str): Path to the dataset directory.
        images_path (str): Path to the images directory.
        labels_path (str): Path to the labels directory.
        images_train_path (str): Path to the training images directory.
        images_validation_path (str): Path to the validation images directory.
        images_test_path (str): Path to the test images directory.
        labels_train_path (str): Path to the training labels directory.
        labels_validation_path (str): Path to the validation labels directory.
        labels_test_path (str): Path to the test labels directory.
        config_path (str): Path to the configuration file.
        split_ratio (dict): Dictionary containing the split ratios for train, validation, and test datasets.
    """

    dataset_path = "./dataset/"
    images_path = f"{dataset_path}/images"
    labels_path = f"{dataset_path}/labels"

    images_train_path = f"{images_path}/train"
    images_validation_path = f"{images_path}/val"
    images_test_path = f"{images_path}/test"

    labels_train_path = f"{labels_path}/train"
    labels_validation_path = f"{labels_path}/val"
    labels_test_path = f"{labels_path}/test"

    config_path = f"{dataset_path}/config.yaml"

    random.seed(42)

    split_ratio = {"train": 0.6, "val": 0.2, "test": 0.2}

    def pre_process(self) -> None:
        """
        Preprocesses the data by splitting it into train, validation, and test sets, and generating a configuration file.
        """
        print("Starting data pre-processing")

        if (
            not os.path.exists(self.dataset_path)
            or not os.listdir(f"{self.dataset_path}/download/images")
            or not os.listdir(f"{self.dataset_path}/download/labels")
        ):
            print("Dataset not found. Download it first")
            return

        if (
            os.path.exists(self.dataset_path)
            and os.path.exists(self.images_path)
            and os.path.exists(self.labels_path)
            and os.listdir(self.images_path)
            and os.listdir(self.labels_path)
        ):
            print("Data already pre-processed")
            return

        # Create processed directory if not exists
        self.ensure_folders_exists()

        # Get all images and labels
        images = glob(f"{DataDownloader.images_path}/*")
        labels = glob(f"{DataDownloader.labels_path}/*.txt")

        # Pair images and labels
        pair_images_labels = list(zip(images, labels))
        random.shuffle(pair_images_labels)

        # Split data
        train_index = int(len(pair_images_labels) * self.split_ratio["train"])
        validation_index = int(
            len(pair_images_labels)
            * (self.split_ratio["train"] + self.split_ratio["val"])
        )

        train_data = pair_images_labels[:train_index]
        validation_data = pair_images_labels[train_index:validation_index]
        test_data = pair_images_labels[validation_index:]

        # Copy images and labels to the right folder
        self.copy_images_and_labels(
            train_data, self.images_train_path, self.labels_train_path
        )
        self.copy_images_and_labels(
            validation_data, self.images_validation_path, self.labels_validation_path
        )
        self.copy_images_and_labels(
            test_data, self.images_test_path, self.labels_test_path
        )

        # Generate data.yaml file
        self.generate_yaml_file(self.config_path)
        print("Data pre-processing finished")

    def ensure_folders_exists(self) -> None:
        """
        Ensures that the necessary directories for storing processed data exist.
        """
        os.makedirs(self.dataset_path, exist_ok=True)

        os.makedirs(self.images_train_path, exist_ok=True)
        os.makedirs(self.images_validation_path, exist_ok=True)
        os.makedirs(self.images_test_path, exist_ok=True)

        os.makedirs(self.labels_train_path, exist_ok=True)
        os.makedirs(self.labels_validation_path, exist_ok=True)
        os.makedirs(self.labels_test_path, exist_ok=True)

    def copy_images_and_labels(self, data, images_path, labels_path) -> None:
        """
        Copies images and labels to the specified directories.

        Args:
            data (list): List of tuples containing image and label file paths.
            images_path (str): Path to the directory where images will be copied.
            labels_path (str): Path to the directory where labels will be copied.
        """
        for image, label in data:
            image_name = os.path.basename(image)
            label_name = os.path.basename(label)

            image_dst = f"{images_path}/{image_name}"
            label_dst = f"{labels_path}/{label_name}"

            shutil.copy(image, image_dst)
            shutil.copy(label, label_dst)

    def generate_yaml_file(self, result_path) -> None:
        """
        Generates a YAML configuration file for the dataset.

        Args:
            result_path (str): Path to the resulting YAML file.
        """
        data = {
            "train": "./images/train",
            "val": "./images/val",
            "test": "./images/test",
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
