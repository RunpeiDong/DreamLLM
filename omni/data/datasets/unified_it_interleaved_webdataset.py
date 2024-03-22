import random
from typing import Callable

import webdataset as wds

from omni.utils.loguru import logger

from ..manager.dataset_info import WebDatasetInfo
from ..manager.dataset_type import InterleavedImageTextReturnType
from .base_dataset import InterleavedImageTextDataset


def filter_no_text_or_no_image(sample):
    return (b"text_list" in sample["json"]) and (b"image_info" in sample["json"])


class UnifiedInterleavedITWebdataset(InterleavedImageTextDataset):
    def __init__(self, dataset_info: WebDatasetInfo, **kwargs):
        """
        A unified Interleaved Image Text Pair Webdataset.

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
            wds.shuffle(1000, rng=random.Random(self.seed), handler=wds.warn_and_continue),
            wds.select(filter_no_text_or_no_image),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.interleaved_to_dict("json;jpg;png;jpeg", handler=wds.warn_and_continue),
            wds.map(self.to_return_type, handler=wds.warn_and_continue),
        )

        self.inner_iter = iter(self.data_pipeline)

        logger.info(
            f"Interleaved Image-Text WebDataset {self.dataset_info.name} loaded. Approximate total number of samples: {self.dataset_info.approx_size}"
        )

    def to_return_type(self, sample) -> InterleavedImageTextReturnType:
        text_list = sample["json"]["text_list"]
        image_list = []
        matched_text_index = []
        matched_sim = []
        for _image_info in sample["json"]["image_info"]:
            image_name = _image_info["image_name"].split(".")[0] + ".jpg"  # images are restored as jpg now
            image_list.append(sample[image_name])
            matched_text_index.append(_image_info["matched_text_index"])
            if "matched_sim" in _image_info.keys():
                matched_sim.append(_image_info["matched_sim"])
            else:
                matched_sim.append(None)
        sequential_index = range(len(matched_text_index))
        zipped_lists = zip(matched_text_index, sequential_index, image_list, matched_sim)
        sorted_pairs = sorted(zipped_lists)
        matched_text_index, _, image_list, matched_sim = zip(*sorted_pairs)
        matched_text_index = list(matched_text_index)
        image_list = list(image_list)
        matched_sim = list(matched_sim)
        return InterleavedImageTextReturnType(
            dataset_type=self.dataset_type,
            image_list=image_list,
            text_list=text_list,
            matched_text_index=matched_text_index,
            matched_sim=matched_sim,
        )

    def __len__(self):
        return self.dataset_info.approx_size

    def __getitem__(self, idx) -> InterleavedImageTextReturnType:
        return next(self.inner_iter)
