import argparse

from datetime import datetime
from picsellia.types.enums import ExperimentStatus

from config import settings
from src.data_downloader import DataDownloader
from src.data_postprocessor import DataPostprocessor
from src.data_preprocessor import DataPreprocessor
from src.data_trainer import DataTrainer
from src.inference import InferenceMode, ImageProcessor
from src.picsellia_connector import PicselliaConnector


def train():
    """
    Trains the model by performing the following steps:
    1. Logs into Picsellia.
    2. Creates a new experiment.
    3. Downloads the dataset and annotations.
    4. Preprocesses the data.
    5. Trains the model.
    6. Evaluates and saves the model to Picsellia.
    """
    dataset_uuid = settings.get("dataset_uuid")

    # Picsellia login
    picselliaConnector = PicselliaConnector()
    client = picselliaConnector.get_client()
    project = picselliaConnector.get_project()
    model = picselliaConnector.get_model()

    experiment_name = "training_" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    project_experiments = project.list_experiments()
    for experiment in project_experiments:
        if experiment.name == experiment_name:
            experiment.delete()

    experiment = project.create_experiment(name=experiment_name)
    experiment.attach_dataset(
        dataset_uuid,
        client.get_dataset_version_by_id(dataset_uuid),
    )

    dataset = client.get_dataset_version_by_id(dataset_uuid)

    # PHASE 1 : Download data and annotations
    data_downloader = DataDownloader(client, dataset)
    data_downloader.download()
    #
    # PHASE 2 : Preprocessing
    data_preprocessor = DataPreprocessor()
    data_preprocessor.pre_process()

    # PHASE 3 : Training
    data_training = DataTrainer(experiment)
    data_training.train()

    # PHASE 4 : Post-processing
    data_postprocessor = DataPostprocessor(experiment, data_training.model)
    data_postprocessor.eval()
    data_postprocessor.save()
    data_postprocessor.upload_to_picsellia(model)

    experiment.update(status=ExperimentStatus.SUCCESS)


def infer_webcam():
    """
    Performs inference using the webcam.
    """
    imageProcessor = ImageProcessor()
    imageProcessor.start_inference(InferenceMode.WEBCAM)


def infer_image(path: str):
    """
    Performs inference on a given image.

    Args:
        path (str): The path to the image file.
    """
    imageProcessor = ImageProcessor()
    imageProcessor.start_inference(InferenceMode.IMAGE, path)


def infer_video(path: str):
    """
    Performs inference on a given video.

    Args:
        path (str): The path to the video file.
    """
    imageProcessor = ImageProcessor()
    imageProcessor.start_inference(InferenceMode.VIDEO, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script pour entraîner ou exécuter l'inférence d'un modèle."
    )

    parser.add_argument(
        "--train", action="store_true", help="Lancer l'entraînement du modèle"
    )

    # Inference options
    parser.add_argument(
        "--infer", choices=["webcam", "image", "video"], help="Mode d'inférence"
    )
    parser.add_argument(
        "--path", type=str, help="Chemin du fichier (requis pour image et vidéo)"
    )

    args = parser.parse_args()

    if args.train:
        train()
    elif args.infer:
        if args.infer == "webcam":
            infer_webcam()
        elif args.infer == "image":
            if args.path:
                infer_image(args.path)
            else:
                print("Erreur: --path est requis pour l'inférence sur une image.")
        elif args.infer == "video":
            if args.path:
                infer_video(args.path)
            else:
                print("Erreur: --path est requis pour l'inférence sur une vidéo.")
    else:
        print("Veuillez spécifier --train ou --infer [webcam/image/video].")
