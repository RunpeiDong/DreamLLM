import json
from dataclasses import dataclass, field

import megfile

from omni.utils.loguru import logger

from ..utils import LargeInt
from .dataset_type import DatasetType


@dataclass
class DatasetInfo:
    name: str | None
    description: str | None = None
    dataset_type: DatasetType = DatasetType.Undefined
    cls: type | str | None = None
    approx_size: str | int | LargeInt | None = None

    def __post_init__(self):
        if self.name is None:
            raise ValueError(f"The `name` of the dataset must be provided.")
        if self.description is None:
            raise ValueError(f"The `description` of the dataset `{self.name}` must be provided.")
        if self.dataset_type == DatasetType.Undefined:
            raise ValueError(f"The `dataset_type` of the dataset `{self.name}` must be provided.")
        if self.cls is None:
            raise ValueError(f"The `cls` of the dataset `{self.name}` must be provided.")
        if not isinstance(self.cls, type):
            raise TypeError(f"The `cls` of the dataset `{self.name}` must be a class, but got `{type(self.cls)}`")
        if self.approx_size is None:
            raise ValueError(f"The `approx_size` of the dataset `{self.name}` is not provided.")
        if not isinstance(self.approx_size, LargeInt):
            self.approx_size = LargeInt(self.approx_size)


@dataclass
class WebDatasetInfo(DatasetInfo):
    shard_list_path: str | None = None
    shard_list: list[str] | None = None
    json_caption_key: str | None = None

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.shard_list_path is not None or self.shard_list is not None
        ), "Either `shard_list_path` or `shard_list` should be provided."

        if self.shard_list_path is not None:
            if self.shard_list is not None:
                logger.warning("Both `shard_list_path` and `shard_list` are provided. `shard_list` will be ignored.")
            self.shard_list = json.load(open(self.shard_list_path, "r"))


@dataclass
class JsonDatasetInfo(DatasetInfo):
    json_list: list[str] | None = None
    json_path: str | None = None
    root: str = None

    def __post_init__(self):
        super().__post_init__()
        assert self.json_list is not None or self.json_path is not None, "Either `json_list` or `json_dir` should be provided."
        if self.json_path is not None:
            if self.json_list is not None:
                logger.warning("Both `json_list` and `json_dir` are provided. `json_list` will be ignored.")
            if megfile.smart_isdir(self.json_path):
                self.json_list = megfile.smart_glob(f"{self.json_path}/*.json")
            else:
                self.json_list = [self.json_path]


@dataclass
class HFITDatasetInfo(DatasetInfo):
    format: str = field(default="json", metadata={"help": "json, cvs, parquet, text, arrow, etc."})
    data_files: str | list[str] | dict[str, list[str] | str] = None
    image_column: str | None = None
    text_column: str | None = None

    def __post_init__(self):
        super().__post_init__()
        assert self.data_files is not None, "`data_files` should be provided."
        assert self.image_column is not None, "`image_column` should be provided."
        assert self.text_column is not None, "`text_column` should be provided."


@dataclass
class SimpleITDatasetInfo(DatasetInfo):
    data_dir: str | None = None

    def __post_init__(self):
        super().__post_init__()
        assert self.data_dir is not None, "`data_dir` should be provided."
        if self.data_dir[-1] == "/":
            self.data_dir = self.data_dir[:-1]


@dataclass
class WebVidDatasetInfo(DatasetInfo):
    shard_list: list[str] | None = None
    data_dir: str | None = None

    def __post_init__(self):
        super().__post_init__()
        assert self.shard_list is not None, "`shard_list` should be provided."
        assert self.data_dir is not None, "`data_dir` should be provided."
