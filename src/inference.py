from ultralytics import YOLO
from enum import Enum

from src.utils import get_model_device


class InferenceMode(Enum):
    WEBCAM = 0
    IMAGE = 1
    VIDEO = 2


def start_inference(
    mode: InferenceMode = InferenceMode.WEBCAM, path: str = None
) -> None:
    model = YOLO("../weights/best.pt")

    if mode == InferenceMode.WEBCAM:
        model(0, device=get_model_device(), show=True)
    elif mode == InferenceMode.IMAGE:
        result = model(path, device=get_model_device())
        result[0].show()
    elif mode == InferenceMode.VIDEO:
        model(path, device=get_model_device(), show=True)
