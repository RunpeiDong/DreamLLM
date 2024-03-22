import random
from typing import Callable

import webdataset as wds

from omni.utils.loguru import logger

from ..manager.dataset_info import WebDatasetInfo
from ..manager.dataset_type import ImageTextTokenPairReturnType
from .base_dataset import ImageTextTokenPairDataset


class UnifiedITTokenPairWebdataset(ImageTextTokenPairDataset):
    def __init__(self, dataset_info: WebDatasetInfo, **kwargs):
        """
        A unified Image Text Pair Webdataset.

        Args:
            dataset_info (WebDatasetInfo): The dataset info.
            image_processor (Callable, optional): Post process image. Defaults to `lambda x: x`.
            text_processor (Callable, optional): Post process text. Defaults to `lambda x: x`.
            seed (int, optional): Seed used to scramble the dataset. Defaults to 42.
        """
        super().__init__(dataset_info)
        self.image_processor: Callable = kwargs.get("image_processor", lambda x: x)
        self.text_processor: Callable = kwargs.get("text_processor", lambda x: x)
        self.seed: int = kwargs.get("seed", 42)

        self.data_pipeline = wds.DataPipeline(
            wds.ResampledShards(self.dataset_info.name, self.dataset_info.shard_list),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue, rng=random.Random(42)),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg;png;jpeg", "txt", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.image_processor, self.text_processor, lambda x: x, handler=wds.warn_and_continue),
            wds.map(self.to_return_type, handler=wds.warn_and_continue),
        )

        self.inner_iter = iter(self.data_pipeline)

        logger.info(
            f"Pretokenized Image-Text Pair WebDataset {self.dataset_info.name} loaded. Approximate total number of samples: {self.dataset_info.approx_size}."
        )

    def to_return_type(self, sample) -> ImageTextTokenPairReturnType:
        return ImageTextTokenPairReturnType(dataset_type=self.dataset_type, image=sample[0], text=sample[1], info=sample[2])

    def __len__(self):
        return self.dataset_info.approx_size

    def __getitem__(self, idx) -> ImageTextTokenPairReturnType:
        return next(self.inner_iter)
