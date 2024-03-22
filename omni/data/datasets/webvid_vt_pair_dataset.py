import os
import random

import isodate
import jsonlines
import megfile

from omni.data.manager.dataset_type import VideoTextPairReturnType
from omni.utils.loguru import logger
from omni.utils.video_utils import load_video

from ..manager.dataset_info import WebVidDatasetInfo
from .base_dataset import VideoTextPairDataset


class WebVidVTPairDataset(VideoTextPairDataset):
    def __init__(self, dataset_info: WebVidDatasetInfo, **kwargs):
        super().__init__(dataset_info)
        self.frames: int = kwargs.get("frames", 4)
        self.seed: int = kwargs.get("seed", 42)

        self.rng = random.Random(self.seed)
        self.shard_list = dataset_info.shard_list
        self.data_dir = dataset_info.data_dir
        self.inner_iter = self._inner_iter(self.rng, self.shard_list, self.data_dir)

        logger.info(
            f"Video-Text Pair Dataset {self.dataset_info.name} loaded. Approximate total number of samples: {self.dataset_info.approx_size}. Frames: {self.frames}"
        )

    def _inner_iter(self, rng: random.Random, shard_list: list[str], data_dir: str):
        while True:
            index = rng.randint(0, len(shard_list) - 1)
            infos = jsonlines.Reader(megfile.smart_open(shard_list[index], "r"))
            for info in infos:
                path = os.path.join(data_dir, info["page_dir"], f"{info['videoid']}.mp4")
                if not megfile.smart_isfile(path):
                    continue
                text = info["name"]
                seconds = isodate.parse_duration(info["duration"]).total_seconds()
                yield (path, text, seconds)

    def __len__(self):
        return self.dataset_info.approx_size

    def __getitem__(self, index) -> VideoTextPairReturnType:
        path, text, seconds = next(self.inner_iter)
        video, video_info = load_video(
            path,
            num_frames=self.frames,
            output_type="pil",
            return_info=True,
        )
        if abs(video_info["duration"] - seconds) > 1:
            logger.warning(f"Video duration mismatch: {video_info['duration']} != {seconds}")
        return VideoTextPairReturnType(dataset_type=self.dataset_type, video=video, video_info=video_info, text=text)
