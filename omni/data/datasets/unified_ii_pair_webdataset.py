import random
from functools import partial
from typing import Callable

import webdataset as wds

from omni.utils.loguru import logger

from ..manager.dataset_info import WebDatasetInfo
from ..manager.dataset_type import ImageImagePairReturnType
from .base_dataset import ImageImagePairDataset
from .unified_it_pair_webdataset import filter_no_caption_or_no_image, filter_none, filter_size


class UnifiedIIPairWebdataset(ImageImagePairDataset):
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
        self.min_size: int = kwargs.get("min_size", -1)
        self.image_processor: Callable = kwargs.get("image_processor", lambda x: x)
        self.text_processor: Callable = kwargs.get("text_processor", lambda x: x)
        self.seed: int = kwargs.get("seed", 42)

        self.data_pipeline = wds.DataPipeline(
            wds.ResampledShards(self.dataset_info.name, self.dataset_info.shard_list),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, rng=random.Random(self.seed), handler=wds.warn_and_continue),
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            # BUG: "json;txt" could cause dataloader lock
            # wds.to_tuple("jpg;png;jpeg", "json;txt", handler=wds.warn_and_continue),
            wds.to_tuple("jpg;png;jpeg", "txt", handler=wds.warn_and_continue),
            wds.map(self.unify_file_format, handler=wds.warn_and_continue),
            wds.select(filter_none),
            wds.select(partial(filter_size, min_size=self.min_size)),
            wds.map_tuple(self.image_processor, self.text_processor, handler=wds.warn_and_continue),
            wds.map(self.to_return_type, handler=wds.warn_and_continue),
        )

        self.inner_iter = iter(self.data_pipeline)

        logger.info(
            f"Image-Image Pair WebDataset {self.dataset_info.name} loaded. Approximate total number of samples: {self.dataset_info.approx_size}. Filter size: {self.min_size}"
        )

    def unify_file_format(self, sample):
        if isinstance(sample[1], dict) and "caption" in sample[1].keys():
            return (sample[0], sample[1]["caption"])
        return sample

    def to_return_type(self, sample) -> ImageImagePairReturnType:
        return ImageImagePairReturnType(
            dataset_type=self.dataset_type, image_source=sample[0], image_target=sample[0], text=sample[1]
        )

    def __len__(self):
        return self.dataset_info.approx_size

    def __getitem__(self, idx) -> ImageImagePairReturnType:
        return next(self.inner_iter)
