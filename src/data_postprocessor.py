from picsellia import Experiment, Model
from picsellia.types.enums import LogType, Framework, InferenceType
from ultralytics. models. yolo. model import YOLO

class DataPostprocessor:
    def __init__(self, experiment: Experiment, yolo: YOLO):
        self.yolo = yolo
        self.experiment = experiment

    def eval(self):
        print("Starting evaluation")
        results = self.yolo.val()

        for key, value in results.results_dict.items():
            self.experiment.log(f"result_{key}", str(value), LogType.VALUE)

    def save_to_picsellia(self, model: Model):
        print("Saving model...")

        self.yolo.save("best.pt")

        labels: dict = {
            str(index): label.name for index, label in
                        enumerate(self.experiment.get_dataset("cnam_products_2024").list_labels())}

        version = model.create_version(
            labels=labels,
            name=self.experiment.name + "_best",
            framework=Framework.PYTORCH,
            type=InferenceType.OBJECT_DETECTION,
        )

        version.store(name="best.pt", path="best.pt", do_zip=True)

        print("Model saved")
