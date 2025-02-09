from picsellia import Experiment, Model
from picsellia.types.enums import LogType, Framework, InferenceType
from ultralytics.models.yolo.model import YOLO
import os


class DataPostprocessor:
    """
    Class to handle the postprocessing of data, including evaluation, saving, and uploading the model.

    Attributes:
        yolo (YOLO): The YOLO model used for evaluation and saving.
        experiment (Experiment): The experiment instance for logging and uploading results.
    """

    def __init__(self, experiment: Experiment, yolo: YOLO):
        """
        Initializes the DataPostprocessor with the given experiment and YOLO model.

        Args:
            experiment (Experiment): The experiment instance.
            yolo (YOLO): The YOLO model instance.
        """
        self.yolo = yolo
        self.experiment = experiment

    def eval(self):
        """
        Evaluates the YOLO model and logs the results to the experiment.
        """
        print("Starting evaluation")
        results = self.yolo.val()

        for key, value in results.results_dict.items():
            self.experiment.log(f"result_{key}", str(value), LogType.VALUE)

    def save(self):
        """
        Saves the YOLO model to a file named 'best.pt'.
        """
        self.yolo.save("best.pt")

    def upload_to_picsellia(self, model: Model):
        """
        Uploads the saved YOLO model to Picsellia.

        Args:
            model (Model): The model instance to which the version will be uploaded.
        """
        print("Saving model...")

        # If the file 'best.pt' does not exist, return
        if not os.path.exists("best.pt"):
            print("Error: best.pt not found")
            return

        self.experiment.list_attached_dataset_versions()[0].list_labels()

        labels: dict = {
            str(index): label.name
            for index, label in enumerate(
                self.experiment.list_attached_dataset_versions()[0].list_labels()
            )
        }

        version = model.create_version(
            labels=labels,
            name=self.experiment.name + "_best",
            framework=Framework.PYTORCH,
            type=InferenceType.OBJECT_DETECTION,
        )

        version.store(name=self.experiment.name + "_best", path="best.pt", do_zip=True)

        print("Model saved")
