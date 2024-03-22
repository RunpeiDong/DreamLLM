import bisect
from typing import Iterable

from torch.utils.data import Dataset, IterableDataset

from omni.utils.loguru import logger

from ..datasets.base_dataset import BaseDataset
from ..utils import LargeInt
from .dataset_type import ReturnType


class MixedDataset(Dataset):
    datasets: list[BaseDataset]
    cumulative_sizes: list[LargeInt]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for l in sequence:
            r.append(l + s)
            s += l
        return r

    def __init__(
        self,
        datasets: Iterable[BaseDataset],
        size_list: list[int | LargeInt],
    ):
        super().__init__()
        self.datasets = datasets
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.size_list = size_list
        assert len(self.datasets) == len(self.size_list), "The length of `datasets` and `size_list` should be the same."
        for dataset, size in zip(self.datasets, self.size_list):
            if size > len(dataset):
                logger.warning(
                    f"Size {size} is larger than dataset length {LargeInt(len(dataset))}, dataset `{dataset.dataset_info.name}` **might** be repeatedly sampled in an epoch."
                )
        self.cumulative_sizes = self.cumsum(self.size_list)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx) -> ReturnType:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx % len(self.datasets[dataset_idx])]
