import argparse
import os
from pathlib import Path
from uuid import UUID

from pandas.core.interchange.dataframe_protocol import DataFrame
from picsellia import Client
from picsellia.types.enums import InferenceType, JobStatus, ExperimentStatus

from config import settings
from src.data_downloader import DataDownloader
from src.data_postprocessor import DataPostprocessor
from src.data_preprocessor import DataPreprocessor
from src.data_trainer import DataTrainer
from src.inference_webcam import InferenceWebcam

def train():
    organization_id = UUID(settings.get("organization_id"))
    api_token = settings.get("api_token")
    dataset_uuid = settings.get("dataset_uuid")
    project_name = settings.get("project_name")

    # Picsellia login
    client = Client(api_token=api_token, organization_id=organization_id)
    project = client.get_project(project_name)

    # Experiment management
    # For now, we will just create one experiment and reset it over and over.
    # Later we may want to have multiple experiments for hyperparameters comparison purposes I guess.
    experiment_name = "experiment"
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

    experiment.update(status=ExperimentStatus.SUCCESS)


def infer():
    # Inference
    inference = InferenceWebcam()
    inference.start_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script pour entraîner ou exécuter l'inférence d'un modèle.")

    parser.add_argument("--train", action="store_true", help="Lancer l'entraînement du modèle")
    parser.add_argument("--infer", action="store_true", help="Lancer l'inférence")

    args = parser.parse_args()

    if args.train:
        train()
    elif args.infer:
        infer()
    else:
        print("Veuillez spécifier --train ou --infer")


