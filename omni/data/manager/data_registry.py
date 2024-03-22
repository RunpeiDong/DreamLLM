from omegaconf import DictConfig

from omni.config.instantiate import deep_instantiate
from omni.config.registry import _convert_target_to_string, locate
from omni.utils.loguru import logger

from ..utils import LargeInt
from .dataset_info import DatasetInfo
from .mixed_dataset import MixedDataset


class DataRegistry:
    def __init__(self, registry_name: str):
        self.__registry_name = registry_name
        self.__registry: dict[str, DatasetInfo] = {}

    def _register(self, info: DatasetInfo | DictConfig, force: bool = False):
        if not isinstance(info, (DatasetInfo, DictConfig)):
            raise TypeError(f"The registration object must be a `DatasetInfo` or `DictConfig`, but got {type(info)}")

        if info.name in self.__registry:
            if not force:
                raise KeyError(f"`{info.name}` is already registered in `{self.__registry_name}`")
            else:
                logger.warning(f"`{info.name}` is already registered in `{self.__registry_name}`, but we force to override it.")
        info.cls = _convert_target_to_string(info.cls)
        if isinstance(info, DictConfig):
            logger.warning(
                f"{info.name.ljust(32)} will be lazy registered, only when using the dataset can one know if there is any incorrect information in the dataset."
            )
        self.__registry[info.name] = info

    def register(self, info: DatasetInfo | list[DatasetInfo] | DictConfig | list[DictConfig], force: bool = False):
        if isinstance(info, (DatasetInfo, DictConfig)):
            info = [info]
        for _info in info:
            try:
                self._register(_info, force=force)
            except:
                logger.warning(f"{info.name} is not registered in {self.__registry_name}.")

    def __repr__(self) -> str:
        # repr = ""
        # for k, v in self.__registry.items():
        #     repr += f"    {k.ljust(20)}, {str(v.approx_size).ljust(6)}, {v.description}\n"
        # return f"{self.__registry_name}(\n{repr})"
        repr = ""
        repr += "-" * 116 + "\n"
        repr += f"| {'Dataset Name'.ljust(32)}| {'Size'.ljust(8)}| {'Description'.ljust(68)} |\n"
        repr += "|" + "-" * 114 + "|" + "\n"
        for k, v in self.__registry.items():
            repr += f"| {k.ljust(32)}| {str(v.approx_size).ljust(8)}| {v.description[:64].ljust(64)} ... |\n"
        repr += "-" * 116 + "\n"
        return repr

    def __call__(
        self,
        datasets: list[str],
        datasets_init_kwargs: dict = {},
        size_list: list[str | int | LargeInt] | None = None,
        ratio: list[float | int] | None = None,
        total_size: str | int | LargeInt | None = None,
    ) -> MixedDataset:
        size_mode = size_list is not None and len(size_list) != 0
        ratio_mode = (ratio is not None and len(ratio) != 0) and total_size is not None
        if size_mode and ratio_mode:
            logger.warning(
                f"Both `size_list` and (`ratio`, `total_size`) are provided. (`ratio`, `total_size`) will be ignored."
            )
        if not (size_mode or ratio_mode):
            logger.warning(f"Neither `size_list` nor (`ratio`, `total_size`) are provided, directly concat the datasets.")
            size_list = [0] * len(datasets)

        if ratio_mode:
            ratio = [1.0 * size / total_size for size in size_list]
            total_size = LargeInt(total_size)
            size_list = [LargeInt(total_size * _ratio) for _ratio in ratio]

        size_list = [LargeInt(size) for size in size_list]

        assert len(datasets) == len(size_list), "The length of `datasets` and `size_list` should be the same."

        iterable_datasets = []
        for name in datasets:
            try:
                info = self.__registry[name]
                if isinstance(info, DictConfig):
                    info.cls = locate(info.cls)
                    info = deep_instantiate(info)
                    info.cls = _convert_target_to_string(info.cls)
            except KeyError:
                logger.info(f"Available datasets:\n{self}")
                raise KeyError(f"`{name}` is not registered in `{self.__registry_name}`")
            cls = locate(info.cls)
            iterable_datasets.append(cls(info, **datasets_init_kwargs))

        if all(size == 0 for size in size_list):
            size_list = [len(_dataset) for _dataset in iterable_datasets]

        return MixedDataset(iterable_datasets, size_list=size_list)
