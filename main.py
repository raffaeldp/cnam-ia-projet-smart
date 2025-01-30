import os
from uuid import UUID

from pandas.core.interchange.dataframe_protocol import DataFrame
from picsellia import Client
from picsellia.types.enums import InferenceType

from config import settings
from src.data_downloader import DataDownloader


def main():
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

    # PHASE 1 : Récupération dataset / annotations YOLO
    data_downloader = DataDownloader(client, dataset)
    data_downloader.download()


    # dataset.set_type(type=InferenceType.OBJECT_DETECTION)
    # dataset.import_annotations_yolo_files("", "./downloads/annotations")
    # assets = dataset.list_assets()
    #
    # assets.download("./downloads/assets")

if __name__ == "__main__":
    main()

