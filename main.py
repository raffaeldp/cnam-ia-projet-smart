import argparse
from uuid import UUID

from datetime import datetime
from picsellia import Client
from picsellia.types.enums import ExperimentStatus

from config import settings
from src.data_downloader import DataDownloader
from src.data_postprocessor import DataPostprocessor
from src.data_preprocessor import DataPreprocessor
from src.data_trainer import DataTrainer
from src.inference import start_inference, InferenceMode


def train() -> None:
    organization_id = UUID(settings.get("organization_id"))
    api_token = settings.get("api_token")
    dataset_uuid = settings.get("dataset_uuid")
    project_name = settings.get("project_name")

    # Picsellia login
    client = Client(api_token=api_token, organization_id=organization_id)
    project = client.get_project(project_name)
    model = client.get_model_by_id(settings.get("model_id"))

    experiment_name = "training_" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
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
    data_postprocessor.save_to_picsellia(model)

    experiment.update(status=ExperimentStatus.SUCCESS)


def infer_webcam() -> None:
    # Inference
    start_inference(InferenceMode.WEBCAM)


def infer_image(path: str) -> None:
    start_inference(InferenceMode.IMAGE, path)


def infer_video(path: str) -> None:
    start_inference(InferenceMode.VIDEO, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script pour entraîner ou " + "exécuter l'inférence d'un modèle.."
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
