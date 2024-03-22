from torch.utils.data import Dataset

from ..manager.dataset_info import DatasetInfo
from ..manager.dataset_type import (
    ConversationReturnType,
    DatasetType,
    ImageImagePairReturnType,
    ImageTextPairReturnType,
    ImageTextTokenPairReturnType,
    InterleavedImageTextReturnType,
    InstructInterleavedImageTextReturnType,
    VideoReturnType,
    VideoTextPairReturnType,
)


class BaseDataset(Dataset):
    dataset_info: DatasetInfo | None = None
    __dataset_type: DatasetType = DatasetType.Undefined

    def __init__(self, dataset_info: DatasetInfo):
        super().__init__()
        assert dataset_info is not None, "`dataset_info` should not be `None`."
        self.dataset_info = dataset_info
        assert (
            self.dataset_info.dataset_type == self.dataset_type
        ), f"Dataset type mismatch: {self.dataset_info.dataset_type} != {self.dataset_type}"

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type


class ImageTextPairDataset(BaseDataset):
    __dataset_type: DatasetType = DatasetType.ImageTextPair

    def __getitem__(self, index) -> ImageTextPairReturnType:
        raise NotImplementedError(
            "This method should be implemented in the subclass with return type `ImageTextPairReturnType`."
        )

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type


class ImageTextTokenPairDataset(BaseDataset):
    __dataset_type: DatasetType = DatasetType.ImageTextTokenPair

    def __getitem__(self, index) -> ImageTextTokenPairReturnType:
        raise NotImplementedError(
            "This method should be implemented in the subclass with return type `ImageTextTokenPairReturnType``"
        )

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type


class ImageImagePairDataset(BaseDataset):
    __dataset_type: DatasetType = DatasetType.ImageImagePair

    def __getitem__(self, index) -> ImageImagePairReturnType:
        raise NotImplementedError(
            "This method should be implemented in the subclass with return type `ImageTextPairReturnType`."
        )

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type


class InterleavedImageTextDataset(BaseDataset):
    __dataset_type: DatasetType = DatasetType.InterleavedImageText

    def __getitem__(self, index) -> InterleavedImageTextReturnType:
        raise NotImplementedError(
            "This method should be implemented in the subclass with return type `InterleavedImageTextReturnType`."
        )

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type


class InstructInterleavedImageTextDataset(BaseDataset):
    __dataset_type: DatasetType = DatasetType.InstructInterleavedImageText

    def __getitem__(self, index) -> InstructInterleavedImageTextReturnType:
        raise NotImplementedError(
            "This method should be implemented in the subclass with return type `InterleavedImageTextReturnType`."
        )

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type


class ConversationDataset(BaseDataset):
    __dataset_type: DatasetType = DatasetType.Conversation

    def __getitem__(self, index) -> ConversationReturnType:
        raise NotImplementedError(
            "This method should be implemented in the subclass with return type `ConversationReturnType`."
        )

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type


class VideoDataset(BaseDataset):
    __dataset_type: DatasetType = DatasetType.Video

    def __getitem__(self, index) -> VideoReturnType:
        raise NotImplementedError("This method should be implemented in the subclass with return type `VideoReturnType`.")

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type


class VideoTextPairDataset(BaseDataset):
    __dataset_type: DatasetType = DatasetType.VideoTextPair

    def __getitem__(self, index) -> VideoTextPairReturnType:
        raise NotImplementedError(
            "This method should be implemented in the subclass with return type `VideoTextPairReturnType`."
        )

    @property
    def dataset_type(self) -> DatasetType:
        return self.__dataset_type
