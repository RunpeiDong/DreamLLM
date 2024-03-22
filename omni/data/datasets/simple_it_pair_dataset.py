from typing import Callable

import megfile

from omni.utils.image_utils import load_image
from omni.utils.loguru import logger

from ..manager.dataset_info import SimpleITDatasetInfo
from ..manager.dataset_type import ImageTextPairReturnType
from .base_dataset import ImageTextPairDataset

IMAGE_EXT = ["png", "jpeg", "jpg"]
IMAGE_EXT = IMAGE_EXT + [_ext.upper() for _ext in IMAGE_EXT]


class SimpleITPairDataset(ImageTextPairDataset):
    def __init__(self, dataset_info: SimpleITDatasetInfo, **kwargs):
        """
        A Simple Image Text Pair Dataset, which only contains few images and texts.

        Args:
            dataset_info (SimpleITDatasetInfo): The dataset info.
            image_processor (Callable, optional): Post process image. Defaults to `lambda x: x`.
            text_processor (Callable, optional): Post process text. Defaults to `lambda x: x`.
        """
        super().__init__(dataset_info)
        self.min_size: int = kwargs.get("min_size", -1)
        self.image_processor: Callable = kwargs.get("image_processor", lambda x: x)
        self.text_processor: Callable = kwargs.get("text_processor", lambda x: x)

        self.images = []
        for _ext in IMAGE_EXT:
            self.images.extend(megfile.smart_glob(f"{self.dataset_info.data_dir}/*.{_ext}"))
        self.captions = megfile.smart_glob(f"{self.dataset_info.data_dir}/*.txt")

        self.images = sorted(self.images)
        self.captions = sorted(self.captions)

        for _image, _caption in zip(self.images, self.captions):
            if _image.split("/")[-1].split(".")[0] != _caption.split("/")[-1].split(".")[0]:
                logger.error(
                    f"{_image} and {_caption} does not match. Full list of images and captions:\n{self.images}\n{self.captions}"
                )

        self.images = [load_image(_image) for _image in self.images]

        logger.info(
            f"Image-Text Pair Dataset {self.dataset_info.name} loaded. Total number of samples: {len(self.images)}. Filter size: {self.min_size} (BUG: no filter)"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> ImageTextPairReturnType:
        return ImageTextPairReturnType(
            dataset_type=self.dataset_type,
            image=self.images[idx],
            text=self.captions[idx],
        )
