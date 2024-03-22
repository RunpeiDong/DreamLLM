import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModalType(Enum):
    IMAGE = "image"


@dataclass
class Unimodal:
    """
    A class that manages a list of unimodal content. The position in the `text_list` is determined by `matched_text_index`.
    `content` will be appended to the `text_list` at the position of `matched_text_index`.
    """

    modal_type: ModalType = field(default=ModalType.IMAGE)
    placeholder: str | None = field(default=None)
    content: list[Any] = field(default_factory=list)
    matched_text_index: list[int] = field(default_factory=list)

    def __post_init__(self):
        """
        You only need specify `modal_type`, `placeholder` and `content` when initializing the class.
        """
        if self.modal_type is None:
            raise ValueError("`modal_type` must be specified.")

        if len(self.content) < 0:
            raise ValueError("The length of `content` must be greater than or equal to 0.")
        if len(self.content) > 0:
            if self.placeholder is None:
                raise ValueError("When `content` is not None, `placeholder` must be specified.")

        if len(self.matched_text_index) > 0:
            if len(self.matched_text_index) != len(self.content):
                raise ValueError(
                    f"The length of `matched_text_index` {len(self.matched_text_index)} and the length of `content` {len(self.content)} should be the same."
                )

    def __len__(self):
        return len(self.content)


@dataclass
class MultimodalContent:
    text: str
    text_list: list[str] = field(default_factory=list)
    unimodal_list: list[Unimodal] = field(default_factory=list)
    mm_content_list: list[Any] = field(default_factory=list)
    mm_placeholder_list: list[str] = field(default_factory=list)
    modal_type_list: list[ModalType] = field(default_factory=list)

    def __post_init__(self):
        """
        You only need specify `text` and `unimodal_list` when initializing the class.
        """
        if len(self.unimodal_list) > 0:
            self.split_text()

    def split_text(self):
        """split text into text_list and fill the matched_text_index of unimodal_list."""
        placeholder_list = [unimodal.placeholder for unimodal in self.unimodal_list]
        pattern = "|".join(placeholder_list)
        regex_pattern = rf"{pattern}"

        matches = list(re.finditer(regex_pattern, self.text))
        last_index = 0
        for match in matches:
            self.text_list.append(self.text[last_index : match.start()])
            for unimodal in self.unimodal_list:
                if match.group() == unimodal.placeholder:
                    unimodal.matched_text_index.append(len(self.text_list))
            last_index = match.end()
        self.text_list.append(self.text[last_index:])

        self.mm_content_list = [None] * (len(self.text_list) - 1)
        self.mm_placeholder_list = [None] * (len(self.text_list) - 1)
        self.modal_type_list = [None] * (len(self.text_list) - 1)

        # NOTE: Convert the matching index to a zero-based index.
        for unimodal in self.unimodal_list:
            assert len(unimodal.matched_text_index) == len(
                unimodal.content
            ), f"The length of `matched_text_index` {len(unimodal.matched_text_index)} and the length of `content` {len(unimodal.content)} should be the same."
            unimodal.matched_text_index = [i - 1 for i in unimodal.matched_text_index]
            for index, content in zip(
                unimodal.matched_text_index,
                unimodal.content,
            ):
                self.mm_content_list[index] = content
                self.mm_placeholder_list[index] = unimodal.placeholder
                self.modal_type_list[index] = unimodal.modal_type
        # " " will be preserved in the text_list.
        # self.text_list = [text.strip() for text in self.text_list]

        total_content = sum([len(unimodal) for unimodal in self.unimodal_list])
        if total_content + 1 != len(self.text_list):
            raise ValueError(
                f"The number of multimodal content + 1 `{total_content + 1}` should be the same as the number of text splits `{len(self.text_list)}`."
            )
