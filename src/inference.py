from ultralytics import YOLO
from enum import Enum

from src.utils import get_model_device


class InferenceMode(Enum):
    """
    Enum class to represent different inference modes.

    Attributes:
        WEBCAM (int): Inference mode for webcam input.
        IMAGE (int): Inference mode for image input.
        VIDEO (int): Inference mode for video input.
    """

    WEBCAM = 0
    IMAGE = 1
    VIDEO = 2


def start_inference(
    mode: InferenceMode = InferenceMode.WEBCAM, path: str = None
) -> None:
    """
    Starts the inference process using the specified mode and path.

    Args:
        mode (InferenceMode): The mode of inference (WEBCAM, IMAGE, or VIDEO).
        path (str, optional): The path to the image or video file. Defaults to None.
    """
    model = YOLO("../weights/best.pt")

    if mode == InferenceMode.WEBCAM:
        model(0, device=get_model_device(), show=True)
    elif mode == InferenceMode.IMAGE:
        result = model(path, device=get_model_device())
        result[0].show()
    elif mode == InferenceMode.VIDEO:
        model(path, device=get_model_device(), show=True)
