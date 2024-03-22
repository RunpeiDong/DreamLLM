import random
import re
from functools import partial
from typing import Callable

import webdataset as wds

from omni.utils.loguru import logger

from ..manager.dataset_info import WebDatasetInfo
from ..manager.dataset_type import ImageTextPairReturnType
from .base_dataset import ImageTextPairDataset


def filter_caption_with_blacklist_words(sample):
    caption = sample[1]
    # fmt: off
    blacklist_words = [
        'image unavailable', 'image', '实拍',
        'com', 'jpg', 'pdf', 'jpeg', 'png', 'tiff', 'doc', 'xlsx', 'xlx', 'ppt',
        'svg', '摄', '本报', '记者', '[二手房]', '原创', '页', '【重要】', '(出租) ',
        'getty', 'image', 'shutterstock', 'shutter', '新闻', '微博', '摄影', '新浪', '贴吧',
        '(图)', '@', '转发', '转载', '抽奖', '赌博', '澳门', '【', '】', '图片', '正文', '回复',
        '淘宝', '拼多多', '微信', '查看源网页', '煎蛋', '果壳'
    ]
    # fmt: on
    for words in blacklist_words:
        if words in caption:
            return False
    return True


def filter_caption_with_only_non_en_words(sample):
    caption = sample[1]
    pattern = re.compile(r"[^\x00-\x7F]+")
    if pattern.search(caption):
        return False
    else:
        return True


def filter_url_caption(sample):
    caption = sample[1]
    url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    if url_pattern.search(caption) is not None:
        return False
    else:
        return True


def filter_caption_with_special_token(sample):
    pass


def filter_no_caption_or_no_image(sample):
    return (("txt" in sample) or ("json" in sample)) and ("png" in sample or "jpg" in sample or "jpeg" in sample)


def filter_none(sample):
    if sample is None:
        return False
    else:
        return sample[0] is not None and sample[1] is not None and sample is not None


def filter_size(sample, min_size: int = 1):
    width, height = sample[0].size
    return width > min_size and height > min_size


class UnifiedITPairWebdataset(ImageTextPairDataset):
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

        if self.dataset_info.json_caption_key is not None:
            dataformat = ("jpg;png;jpeg", "json")
        else:
            dataformat = ("jpg;png;jpeg", "txt")

        self.data_pipeline = wds.DataPipeline(
            wds.ResampledShards(self.dataset_info.name, self.dataset_info.shard_list),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, rng=random.Random(self.seed), handler=wds.warn_and_continue),
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            # BUG: "json;txt" could cause dataloader lock
            # wds.to_tuple("jpg;png;jpeg", "json;txt", handler=wds.warn_and_continue),
            wds.to_tuple(*dataformat, handler=wds.warn_and_continue),
            wds.map(self.unify_file_format, handler=wds.warn_and_continue),
            wds.select(filter_none),
            # BUG: filter_size will cause batch size mismatch, will fix this later (when min_size>0)
            wds.select(partial(filter_size, min_size=self.min_size)),
            wds.map_tuple(self.image_processor, self.text_processor, handler=wds.warn_and_continue),
            wds.map(self.to_return_type, handler=wds.warn_and_continue),
        )

        self.inner_iter = iter(self.data_pipeline)

        self.consecutive_failures = 0

        logger.info(
            f"Image-Text Pair WebDataset {self.dataset_info.name} loaded. Approximate total number of samples: {self.dataset_info.approx_size}. Filter size: {self.min_size}"
        )

    def unify_file_format(self, sample):
        if isinstance(sample[1], dict):
            if self.dataset_info.json_caption_key in sample[1].keys():
                self.consecutive_failures = 0
                return sample[0], sample[1][self.dataset_info.json_caption_key]
            else:
                self.consecutive_failures += 1
                logger.warning(
                    f"Cannot find the key `{self.dataset_info.json_caption_key}` in the json file. Consecutive failures: {self.consecutive_failures}."
                )
                return None
        elif isinstance(sample[1], str):
            if sample[1] == "":
                self.consecutive_failures += 1
                logger.warning(f"Empty caption. Consecutive failures: {self.consecutive_failures}.")
                return None
            else:
                self.consecutive_failures = 0
                return sample
        else:
            self.consecutive_failures += 1
            logger.warning(f"Unknown caption type. Consecutive failures: {self.consecutive_failures}.")
            return None

    def to_return_type(self, sample) -> ImageTextPairReturnType:
        return ImageTextPairReturnType(dataset_type=self.dataset_type, image=sample[0], text=sample[1])

    def __len__(self):
        return self.dataset_info.approx_size

    def __getitem__(self, idx) -> ImageTextPairReturnType:
        return next(self.inner_iter)
