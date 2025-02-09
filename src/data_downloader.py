import os
import shutil
import zipfile

from config import settings
from picsellia import Client, DatasetVersion
from picsellia.types.enums import AnnotationFileType


class DataDownloader:
    """
    Class to handle downloading datasets and annotations from Picsellia.

    Attributes:
        dataset_download_path (str): Path to the directory where the dataset will be downloaded.
        images_path (str): Path to the directory where images will be stored.
        labels_path (str): Path to the directory where labels will be stored.
    """

    dataset_download_path = "./dataset/download"
    images_path = f"{dataset_download_path}/images"
    labels_path = f"{dataset_download_path}/labels"

    def __init__(self, client: Client, dataset: DatasetVersion) -> None:
        """
        Initializes the DataDownloader with the given client and dataset version.

        Args:
            client (Client): The Picsellia client instance.
            dataset (DatasetVersion): The Picsellia dataset version instance.
        """
        self.client = client
        self.dataset = dataset

    def download(self) -> None:
        """
        Downloads the dataset and annotations if they do not already exist.
        """
        # Create download directory if not exists
        if not os.path.exists(self.dataset_download_path):
            print("Creating datasets directory")
            os.makedirs(self.dataset_download_path)

        if (
            os.listdir(self.dataset_download_path)
            and os.path.exists(self.images_path)
            and os.path.exists(self.labels_path)
        ):
            print("Dataset already downloaded")
            return

        self._download_images()
        self._download_labels()

    def _download_images(self) -> None:
        """
        Downloads the images from the dataset.
        """
        # Create images directory if not exists
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

        if os.listdir(self.images_path):
            print("Images already downloaded")
            return

        print("Downloading images...")
        self.dataset.list_assets().download(self.images_path, use_id=True)
        print("Images downloaded")

    def _download_labels(self) -> None:
        """
        Downloads and extracts the labels from the dataset.
        """
        # Create labels directory if not exists
        if not os.path.exists(self.labels_path):
            os.makedirs(self.labels_path)

        if os.listdir(self.labels_path):
            print("Labels already downloaded")
            return

        print("Downloading labels...")
        self.dataset.export_annotation_file(
            AnnotationFileType.YOLO, self.labels_path, use_id=True
        )

        print("Extracting labels...")
        # The .zip is in annotations_path/organization_id/annotations/*.zip. We need to extract it.
        for root, dirs, files in os.walk(self.labels_path):
            for file in files:
                if file.endswith(".zip"):
                    with zipfile.ZipFile(os.path.join(root, file), "r") as zip_ref:
                        zip_ref.extractall(self.labels_path)
                    print(f"Labels extracted in {self.labels_path}")
                    break

        print("Deleting zip file...")
        shutil.rmtree(
            os.path.join(f"{self.labels_path}/{settings.get('organization_id')}")
        )
        print("Labels downloaded")
