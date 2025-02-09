from uuid import UUID

from picsellia import Client, Project, DatasetVersion, Model
from config import settings


class PicselliaConnector:
    def __init__(self):
        self.__client: Client = Client(
            api_token=settings("api_token"),
            organization_id=UUID(settings.get("organization_id")),
        )

        self.__project_name: str = settings("project_name")
        self.__dataset_id: str = settings("dataset_uuid")
        self.__model_id: str = settings("model_id")

        self.__project: Project = self.__client.get_project(self.__project_name)

    def get_client(self) -> Client:
        return self.__client

    def get_project(self) -> Project:
        return self.__project

    def get_dataset(self) -> DatasetVersion:
        return self.__client.get_dataset_version_by_id(self.__dataset_id)

    # def get_experiment(self) -> Experiment:
    #     try:
    #         old_experiment: Experiment = self.__project.get_experiment(
    #             self.__experiment_name
    #         )
    #         old_experiment.delete()
    #     except ResourceNotFoundError:
    #         log(
    #             f"Could not find existing experiment named {self.__experiment_name}, creating a new one."
    #         )
    #         pass
    #     experiment: Experiment = self.__project.create_experiment(
    #         self.__experiment_name
    #     )
    #     experiment.attach_dataset("Dataset", self.get_dataset())
    #     return experiment

    def get_model(self) -> Model:
        return self.__client.get_model_by_id(self.__model_id)
