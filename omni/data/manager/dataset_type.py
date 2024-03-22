from dataclasses import dataclass
from enum import Enum

from omni.conversation.conversation import Dialog

from ..utils import ImageType, VideoType


class DatasetType(Enum):
    Undefined = "Undefined"
    ImageTextPair = "ImageTextPair"
    ImageTextTokenPair = "ImageTextTokenPair"
    ImageImagePair = "ImageImagePair"
    InterleavedImageText = "InterleavedImageText"
    InstructInterleavedImageText = "InstructInterleavedImageText"
    Conversation = "Conversation"
    Video = "Video"
    VideoTextPair = "VideoTextPair"


@dataclass
class ReturnType:
    dataset_type: DatasetType = DatasetType.Undefined


@dataclass
class ImageTextPairReturnType(ReturnType):
    dataset_type: DatasetType = DatasetType.ImageTextPair
    image: ImageType | None = None
    text: str | None = None


@dataclass
class ImageTextTokenPairReturnType(ReturnType):
    dataset_type: DatasetType = DatasetType.ImageTextTokenPair
    image: ImageType | None = None
    text: str | None = None
    info: dict | None = None


@dataclass
class ImageImagePairReturnType(ReturnType):
    dataset_type: DatasetType = DatasetType.ImageImagePair
    image_source: ImageType | None = None
    image_target: ImageType | None = None
    text: str | None = None


@dataclass
class InterleavedImageTextReturnType(ReturnType):
    dataset_type: DatasetType = DatasetType.InterleavedImageText
    image_list: list[ImageType] | None = None
    text_list: list[str] | None = None
    # NOTE: Ensure that this is in ascending order.
    matched_text_index: list[int] | None = None
    matched_sim: list[float] | None = None


@dataclass
class InstructInterleavedImageTextReturnType(InterleavedImageTextReturnType):
    instruction: str | None = None
    dialog: Dialog | None = None


@dataclass
class ConversationReturnType(ReturnType):
    dataset_type: DatasetType = DatasetType.Conversation
    dialog: Dialog | None = None


@dataclass
class VideoReturnType(ReturnType):
    dataset_type: DatasetType = DatasetType.Video
    video: VideoType | None = None
    video_info: dict | None = None


@dataclass
class VideoTextPairReturnType(ReturnType):
    dataset_type: DatasetType = DatasetType.VideoTextPair
    video: VideoType | None = None
    video_info: dict | None = None
    text: str | None = None
