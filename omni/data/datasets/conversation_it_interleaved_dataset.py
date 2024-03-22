import os
import copy
import random
from typing import Callable

import webdataset as wds

from omni.conversation.conversation import Dialog, Message
from omni.conversation.multimodal import ModalType, MultimodalContent, Unimodal
from omni.utils.image_utils import load_image
from omni.utils.json_utils import load_json_list
from omni.utils.loguru import logger

from ..manager.dataset_info import JsonDatasetInfo
from ..manager.dataset_type import ConversationReturnType, InstructInterleavedImageTextReturnType
from .base_dataset import InstructInterleavedImageTextDataset


def filter_no_text_or_no_image(sample):
    return (b"text_list" in sample["json"]) and (b"image_info" in sample["json"])


class InstructInterleavedITConversationDataset(InstructInterleavedImageTextDataset):
    def __init__(self, dataset_info: JsonDatasetInfo, **kwargs):
        """
        A json-based Interleaved Image Text Pair Dataset.

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

        self.data: list[dict] = load_json_list(dataset_info.json_list, keys=None)

        # for compatibility and uniformity
        self.roles_map = {
            "user": "user",
            "human": "user",
            "assistant": "assistant",
            "gpt": "assistant",
            "obj365": "assistant",
            "vg": "assistant",
        }

        logger.info(
            f"Instruction-following Interleaved Image-Text Dataset {self.dataset_info.name} loaded. Approximate total number of samples: {self.dataset_info.approx_size}"
        )

    def __len__(self):
        return len(self.data)

    def to_return_type(self, sample) -> InstructInterleavedImageTextReturnType:
        text_list = sample["text_list"]
        instruction = sample["instruction"]
        image_list = []
        matched_text_index = []
        matched_sim = []
        for _image_info in sample["image_info"]:
            image_name = _image_info["image_name"]
            image_name = image_name if self.dataset_info.root is None else os.path.join(self.dataset_info.root, image_name)
            try:
                image_list.append(load_image(image_name))
                matched_text_index.append(_image_info["matched_text_index"])
                if "matched_sim" in _image_info.keys():
                    matched_sim.append(_image_info["matched_sim"])
                else:
                    matched_sim.append(None)
            except:
                logger.debug(f"image wrong: {image_name}")
                import ipdb; ipdb.set_trace()
                pass
        sequential_index = range(len(matched_text_index))
        zipped_lists = zip(matched_text_index, sequential_index, image_list, matched_sim)
        sorted_pairs = sorted(zipped_lists)
        matched_text_index, _, image_list, matched_sim = zip(*sorted_pairs)
        matched_text_index = list(matched_text_index)
        image_list = list(image_list)
        matched_sim = list(matched_sim)

        # construct multimodal dialog
        dialog: Dialog = []
        # instruction
        dialog.append(
            Message(
                role=self.roles_map["user"],
                content=MultimodalContent(text=instruction),
            )
        )

        dialog.append(
            Message(
                role=self.roles_map["assistant"],
                content=None,
            )
        )

        return InstructInterleavedImageTextReturnType(
            dataset_type=self.dataset_type,
            image_list=image_list,
            text_list=text_list,
            matched_text_index=matched_text_index,
            matched_sim=matched_sim,
            instruction=instruction,
            dialog=dialog,
        )

    def _merge_text_list(self, text_list, matched_text_index):
        new_text_list = []
        prev_index = 0
        for index in matched_text_index:
            new_text_list.append(" ".join(text_list[prev_index : index + 1]))
            prev_index = index + 1
        if prev_index != len(text_list):
            new_text_list.append(" ".join(text_list[prev_index:]))
        return new_text_list

    def __getitem__(self, index) -> InstructInterleavedImageTextReturnType:
        sample: dict = copy.deepcopy(self.data[index])
        return self.to_return_type(sample)
