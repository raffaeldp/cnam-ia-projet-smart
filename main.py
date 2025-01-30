import os
from uuid import UUID

from picsellia import Client

from config import settings


def main():
    # print("HIII")
    # client = Client(
    #     api_token="06b1b31ea34b81392168385c61b26400263d4b14",
    #     organization_name="Picsalex-MLOps",
    #     host=settings.get("host")
    # )
    #
    # project = client.get_project("documentation-project")
    # experiment = project.get_experiment("exp-0-documentation")
    #
    # datalake = client.get_datalake()
    # dataset = client.get_dataset("my-awesome-dataset").get_version("first")
    # assets = dataset.list_assets()
    # assets.download("./downloads/assets")

    organization_id = UUID(settings.get("organization_id"))
    client = Client(api_token=settings.get("api_token"), organization_id=organization_id)

if __name__ == "__main__":
    main()

