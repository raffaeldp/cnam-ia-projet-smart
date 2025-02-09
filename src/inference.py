import os

from picsellia import Model, ModelVersion, ModelFile
from ultralytics import YOLO
from enum import Enum

import zipfile
from src.picsellia_connector import PicselliaConnector
from src.utils import get_model_device


class InferenceMode(Enum):
    """
    Enum class to represent different inference modes.

    Attributes:
        WEBCAM (int): Inference mode for webcam input.
        IMAGE (int): Inference mode for image input.
        VIDEO (int): Inference mode for video input.
    """

    WEBCAM = 0
    IMAGE = 1
    VIDEO = 2


class ImageProcessor:
    __model_download_directory_path = "./weights"
    __model_zip_file_path = f"{__model_download_directory_path}/best.zip"
    __model_file_path = f"{__model_download_directory_path}/best.pt"
    __model_latest_file_path = f"{__model_download_directory_path}/latest"

    def __select_model_version_idx(self, model: Model) -> int:
        model_versions: list[ModelVersion] = model.list_versions()
        print("Here is the available versions :")
        model_versions_with_files = []
        for model_version in model_versions:
            if model_version.list_files():
                model_versions_with_files.append(model_version)

        if not model_versions_with_files:
            print("Error: No model versions with files found")
            exit(1)

        for index, model_version in enumerate(model_versions):
            if model_version.list_files():
                print(f"{index}: {model_version.name}")
        versionIdx: str = input("Choose a version with it's index: ")

        if not (versionIdx.isdigit() and 0 <= int(versionIdx) < len(model_versions)):
            print(f"Error: Model version {versionIdx} does not exist")
            return self.__select_model_version_idx(model)

        return int(versionIdx)

    def __select_model_file(self, model_version: ModelVersion) -> ModelFile:
        model_files: list[ModelFile] = model_version.list_files()
        print("Here is the available files :")
        for index, model_file in enumerate(model_files):
            print(f"{index}: {model_file.name}")
        fileIdx: str = input("Choose a file with it's index: ")

        if not (fileIdx.isdigit() and 0 <= int(fileIdx) < len(model_files)):
            print(f"Error: Model file {fileIdx} does not exist")
            return self.__select_model_file(model_version)
        return model_files[int(fileIdx)]

    def __download_model(self, model_version: ModelVersion, model_file: ModelFile):
        # Delete best.pt file if it exists
        if os.path.exists(self.__model_file_path):
            os.remove(self.__model_file_path)
        if os.path.exists(self.__model_zip_file_path):
            os.remove(self.__model_zip_file_path)
        if os.path.exists(self.__model_latest_file_path):
            os.remove(self.__model_latest_file_path)

        model_file.download(self.__model_download_directory_path)
        # Unzip the downloaded file
        with zipfile.ZipFile(self.__model_zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.__model_download_directory_path)

        # Delete zip file with shutil
        os.remove(self.__model_zip_file_path)

        # Create a latest file with model_version name and model_file name
        with open(self.__model_latest_file_path, "w") as f:
            f.write(f"{model_version.name}\n{model_file.name}")

    def start_inference(
        self, mode: InferenceMode = InferenceMode.WEBCAM, path: str = None
    ) -> None:
        """
        Starts the inference process using the specified mode and path.

        Args:
            mode (InferenceMode): The mode of inference (WEBCAM, IMAGE, or VIDEO).
            path (str, optional): The path to the image or video file. Defaults to None.
        """
        # Create download folder if not exists
        os.makedirs(self.__model_download_directory_path, exist_ok=True)

        picselliaConnector = PicselliaConnector()
        project_model = picselliaConnector.get_model()
        version_idx = self.__select_model_version_idx(project_model)
        version = project_model.list_versions()[version_idx].name
        model_version: ModelVersion = project_model.get_version(version)
        model_file: ModelFile = self.__select_model_file(model_version)

        # Check that the latest file do not contain the same model version and model file
        if os.path.exists(self.__model_latest_file_path):
            with open(self.__model_latest_file_path, "r") as f:
                latest_model_version_name, latest_model_file_name = f.read().split("\n")
                if (
                    latest_model_version_name == model_version.name
                    and latest_model_file_name == model_file.name
                ):
                    print("Model already downloaded")
                else:
                    self.__download_model(model_version, model_file)
        else:
            self.__download_model(model_version, model_file)

        model = YOLO(self.__model_file_path)

        if mode == InferenceMode.WEBCAM:
            model(0, device=get_model_device(), show=True)
        elif mode == InferenceMode.IMAGE:
            result = model(path, device=get_model_device())
            result[0].show()
        elif mode == InferenceMode.VIDEO:
            model(path, device=get_model_device(), show=True)
