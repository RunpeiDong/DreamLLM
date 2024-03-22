import copy
import os
from typing import NewType

from omni.conversation.conversation import Dialog, Message
from omni.conversation.multimodal import ModalType, MultimodalContent, Unimodal
from omni.data.manager.dataset_type import ConversationReturnType
from omni.utils.image_utils import load_image
from omni.utils.json_utils import load_json_list
from omni.utils.loguru import logger

from ..manager.dataset_info import JsonDatasetInfo
from ..utils import LargeInt
from .base_dataset import ConversationDataset

Placeholder = NewType("Placeholder", str)
ModalTypeStr = NewType("ModalTypeStr", str)
Path = NewType("Path", str)


class ConversationDataset(ConversationDataset):
    def __init__(self, dataset_info: JsonDatasetInfo, **kwargs):
        super().__init__(dataset_info)
        self.compatible_keys = ("image",)
        self.keys = ("conversations",)
        self.data: list[dict] = load_json_list(dataset_info.json_list, keys=self.keys + self.compatible_keys)

        # for compatibility and uniformity
        self.roles_map = {
            "user": "user",
            "human": "user",
            "assistant": "assistant",
            "gpt": "assistant",
            "obj365": "assistant",
            "vg": "assistant",
        }
        logger.info(f"Conversation Dataset {self.dataset_info.name} loaded. Total number of samples: {LargeInt(len(self))}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> ConversationReturnType:
        data: dict = copy.deepcopy(self.data[index])
        conversations = data["conversations"]
        for conv in conversations:
            conv["from"] = self.roles_map[conv["from"]]
        if conversations[0]["from"] != "user":
            conversations = conversations[1:]
        dialog: Dialog = []

        if "image" in data.keys():  # compatible with the old format
            old_placehold = ("<image>", "<dream>")
            for conv in conversations:
                if all(_old_placehold not in conv["value"] for _old_placehold in old_placehold):  # pure text
                    dialog.append(
                        Message(
                            role=conv["from"],
                            content=MultimodalContent(text=conv["value"]),
                        )
                    )
                else:
                    for _old_placehold in old_placehold:
                        if _old_placehold in conv["value"]:
                            image_modal = Unimodal(
                                modal_type=ModalType("image"),
                                placeholder=_old_placehold,
                                content=[
                                    load_image(
                                        data["image"]
                                        if self.dataset_info.root is None
                                        else os.path.join(self.dataset_info.root + data["image"])
                                    )
                                ],
                            )
                            dialog.append(
                                Message(
                                    role=conv["from"],
                                    content=MultimodalContent(text=conv["value"], unimodal_list=[image_modal]),
                                )
                            )

        # BUG: noqa
        else:  # new format
            for conv in conversations:
                if "modal_info" not in conv.keys():
                    dialog.append(
                        Message(
                            role=conv["from"],
                            content=MultimodalContent(text=conv["value"]),
                        )
                    )
                else:
                    modal_info: dict[Placeholder, tuple(ModalTypeStr, list[Path])] = conv["modal_info"]
                    unimodal_list = []
                    for placeholder, modal_type_path_list in modal_info.items():
                        modal_type, path_list = tuple(modal_type_path_list)
                        unimodal_list.append(
                            Unimodal(
                                modal_type=ModalType(modal_type),
                                placeholder=placeholder,
                                content=[
                                    load_image(
                                        path if self.dataset_info.root is None else os.path.join(self.dataset_info.root + path)
                                    )
                                    for path in path_list
                                ],
                            )
                        )
                    dialog.append(
                        Message(
                            role=conv["from"],
                            content=MultimodalContent(text=conv["value"], unimodal_list=unimodal_list),
                        )
                    )

        return ConversationReturnType(dataset_type=self.dataset_type, dialog=dialog)
