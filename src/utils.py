import platform

import torch


def get_model_device() -> str:
    """
    Returns the appropriate device for model training or inference based on the operating system.

    Returns:
        str: 'mps' if the operating system is macOS, otherwise 'cuda'.
    """
    return (
        "mps"
        if platform.system() == "Darwin"
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
