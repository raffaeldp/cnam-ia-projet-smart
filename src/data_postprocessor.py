from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics. models. yolo. model import YOLO

class DataPostprocessor:
    def __init__(self, experiment: Experiment, model: YOLO):
        self.model = model
        self.experiment = experiment

    def eval(self):
        print("Starting evaluation")
        results = self.model.val()

        for key, value in results.results_dict.items():
            self.experiment.log(f"result_{key}", str(value), LogType.VALUE)