import io
import os
import torch
import megfile
import requests
import PIL.Image
import PIL.ImageOps

import numpy as np

from pathlib import Path
from typing import Literal

ImageType = Literal["pil", "numpy", "pt"]


# Copy from diffusers/utils/testing_utils.py
def load_image(image: str | PIL.Image.Image, output_type: ImageType = "pil") -> PIL.Image.Image | np.ndarray | torch.FloatTensor:
    """
    Loads `path` to a PIL Image, Numpy or Torch Tensor.

    Args:
        path (str | PIL.Image.Image): The path to image or PIL Image to convert to the ImageType format.
        output_type (ImageType, optional): "pil", "numpy" or "pt". Defaults to "pil".

    Returns:
        PIL.Image.Image | np.ndarray | torch.FloatTensor: The image in the ImageType format.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif image.startswith("s3"):
            with megfile.smart_open(image, "rb") as f:
                bytes_data = f.read()
            image = PIL.Image.open(io.BytesIO(bytes_data), "r")
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(f"Incorrect path or url, URLs must start with `http://`, `https://` or `s3+[profile]://`, and {image} is not a valid path")
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError("Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image.")

    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    if output_type == "pil":
        image = image
    elif output_type == "numpy":
        image = pil_to_numpy(image)
    elif output_type == "pt":
        image = pil_to_pt(image)
    else:
        raise ValueError(f"output_type must be one of {ImageType}, got {output_type}")

    return image


def save_image(image: PIL.Image.Image | np.ndarray | torch.FloatTensor, path: str, format: str | None = None):
    """
    Save PIL Image to path.

    Args:
        image (PIL.Image.Image): The PIL image to be saved.
        path (str): The path to save the image, support s3.
        format (str, optional): Defaults to None. If set, will save the image with the given format.
    """
    if isinstance(image, np.ndarray):
        image = numpy_to_pil(image)[0]
    elif isinstance(image, torch.FloatTensor):
        image = pt_to_pil(image)[0]
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError("Incorrect format used for image. Should be a PIL image, a NumPy array, or a PyTorch tensor.")

    extension = os.path.splitext(path)[-1][1:]
    if format is not None:
        print(format.__class__)
        assert extension.lower() == format.lower(), f"extension {extension} does not match format {format}"

    if path.startswith("s3"):
        buffered = io.BytesIO()
        image.save(buffered, format=format)

        with megfile.smart_open(path, "wb") as f:
            f.write(buffered.getvalue())
    else:
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        image.save(path, format=format)


def numpy_to_pil(images: np.ndarray) -> list[PIL.Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [PIL.Image.fromarray(image) for image in images]

    return pil_images


def pil_to_numpy(images: PIL.Image.Image | list[PIL.Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert a NumPy image (bs, h, w, c) to a PyTorch tensor (bs, c, h, w).
    """
    if images.ndim == 3:
        images = images[None, ...]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images


def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def pt_to_pil(images: torch.FloatTensor) -> list[PIL.Image.Image]:
    images = pt_to_numpy(images)
    images = numpy_to_pil(images)
    return images


def pil_to_pt(images: PIL.Image.Image | list[PIL.Image.Image]) -> torch.FloatTensor:
    images = pil_to_numpy(images)
    images = numpy_to_pt(images)
    return images


def normalize(images):
    """
    Normalize an image array [0, 1] to [-1, 1].
    """
    return 2.0 * images - 1.0


def denormalize(images):
    """
    Denormalize an image array [-1, 1] to [0, 1].
    """
    return (images / 2 + 0.5).clamp(0, 1)


def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts an image to RGB format.
    """
    image = image.convert("RGB")
    return image
