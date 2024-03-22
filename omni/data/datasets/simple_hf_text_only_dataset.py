import io
import time
from typing import Callable

from datasets import load_dataset
from joblib import Parallel, delayed
from PIL import Image

from omni.utils.loguru import logger

from ..manager.dataset_info import HFITDatasetInfo
from ..manager.dataset_type import ImageTextPairReturnType
from .base_dataset import ImageTextPairDataset


class HFITPairDataset(ImageTextPairDataset):
    def __init__(self, dataset_info: HFITDatasetInfo, **kwargs):
        """
        A Hugging Face Image Text Pair Dataset.

        Args:
            dataset_info (HFITDatasetInfo): The dataset info.
            image_processor (Callable, optional): Post process image. Defaults to `lambda x: x`.
            text_processor (Callable, optional): Post process text. Defaults to `lambda x: x`.
        """
        super().__init__(dataset_info)
        self.min_size: int = kwargs.get("min_size", 1)
        self.image_processor: Callable = kwargs.get("image_processor", lambda x: x)
        self.text_processor: Callable = kwargs.get("text_processor", lambda x: x)

        tic = time.time()
        try:
            inner_data = load_dataset(
                dataset_info.format, data_files=dataset_info.data_files, split="train", keep_in_memory=True
            )
        except:
            inner_data = load_dataset(dataset_info.format, data_files=dataset_info.data_files, split="train")

        def process_data(data):
            image_stream = io.BytesIO(data[dataset_info.image_column]["bytes"])
            image = Image.open(image_stream).convert("RGB")
            text = data[dataset_info.text_column]
            return image, text

        results = Parallel(n_jobs=32)(delayed(process_data)(data) for data in inner_data.to_list())
        self.image_list, self.text_list = zip(*results)

        time_cost = time.time() - tic
        logger.info(f"Loading dataset {self.dataset_info.name} takes {time_cost} seconds")

        logger.info(
            f"Image-Text Pair Dataset {self.dataset_info.name} loaded. Total number of samples: {len(self.image_list)}. Filter size: {self.min_size} (BUG: no filter)"
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx) -> ImageTextPairReturnType:
        return ImageTextPairReturnType(
            dataset_type=self.dataset_type,
            image=self.image_list[idx],
            text=self.text_list[idx],
        )
