import platform

def get_model_device() -> str:
    return "mps" if platform.system() == "Darwin" else "cuda"