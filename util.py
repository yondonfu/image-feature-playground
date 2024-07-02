import torch
import io
import PIL.Image
import base64


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def dict_convert_tensors_to_lists(x: dict) -> dict:
    result = {}
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result


def pil_image_to_b64(image: PIL.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="png")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode(
        "utf-8"
    )
