import os
import shutil
import zipfile

from config import settings
from picsellia import Client, Dataset, DatasetVersion
from picsellia.types.enums import AnnotationFileType


class DataDownloader:
    dataset_path = "./dataset"
    images_path = f"{dataset_path}/images"
    annotations_path = f"{dataset_path}/annotations"

    def __init__(self, client: Client, dataset: DatasetVersion):
        self.client = client
        self.dataset = dataset

    def download(self):
        # Create download directory if not exists
        if not os.path.exists(self.dataset_path):
            print("Creating dataset directory")
            os.makedirs(self.dataset_path)

        if os.listdir(self.dataset_path) and os.path.exists(self.images_path) and os.path.exists(self.annotations_path):
            print("Dataset already downloaded")
            return

        self._download_images()
        self._download_annotations()

    def _download_images(self):
        # Download images if not already downloaded
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
        if not os.listdir(self.images_path):
            print("Downloading images")
            self.dataset.list_assets().download(self.images_path)
        else:
            print("Images already downloaded")

    def _download_annotations(self):
        print("Downloading annotations zip file...")
        if not os.path.exists(self.annotations_path):
            os.makedirs(self.annotations_path)
        if not os.listdir(self.annotations_path):
            self.dataset.export_annotation_file(AnnotationFileType.YOLO, self.annotations_path)

        print("Extracting annotations...")
        # The .zip is in annotations_path/organization_id/annotations/*.zip. We need to extract it.
        for root, dirs, files in os.walk(self.annotations_path):
            for file in files:
                if file.endswith(".zip"):
                    with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                        zip_ref.extractall(self.annotations_path)
                    print(f"Annotations extracted in {self.annotations_path}")
                    break

        print("Deleting zip file...")
        shutil.rmtree(os.path.join(f"{self.annotations_path}/{settings.get('organization_id')}"))


