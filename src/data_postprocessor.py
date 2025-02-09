from picsellia import Experiment, Model
from picsellia.types.enums import LogType, Framework, InferenceType
from ultralytics.models.yolo.model import YOLO


class DataPostprocessor:
    """
    Class to handle the postprocessing of data, including evaluation and saving models to Picsellia.

    Attributes:
        yolo (YOLO): The YOLO model instance.
        experiment (Experiment): The Picsellia experiment instance.
    """

    def __init__(self, experiment: Experiment, yolo: YOLO) -> None:
        """
        Initializes the DataPostprocessor with the given experiment and YOLO model.

        Args:
            experiment (Experiment): The Picsellia experiment instance.
            yolo (YOLO): The YOLO model instance.
        """
        self.yolo = yolo
        self.experiment = experiment

    def eval(self) -> None:
        """
        Evaluates the YOLO model and logs the results to Picsellia.
        """
        print("Starting evaluation")
        results = self.yolo.val()

        for key, value in results.results_dict.items():
            self.experiment.log(f"result_{key}", str(value), LogType.VALUE)

    def save_to_picsellia(self, model: Model) -> None:
        """
        Saves the YOLO model to Picsellia.

        Args:
            model (Model): The Picsellia model instance.
        """
        print("Saving model...")

        self.yolo.save("best.pt")

        labels: dict = {
            str(index): label.name
            for index, label in enumerate(
                self.experiment.get_dataset("cnam_products_2024").list_labels()
            )
        }

        version = model.create_version(
            labels=labels,
            name=self.experiment.name + "_best",
            framework=Framework.PYTORCH,
            type=InferenceType.OBJECT_DETECTION,
        )

        version.store(name="best.pt", path="best.pt", do_zip=True)

        print("Model saved")
