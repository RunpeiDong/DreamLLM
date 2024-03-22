import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from omni.constants import IGNORE_INDEX
from omni.conversation.conversation import SeparatorStyle, get_conv_template
from omni.data.utils import LargeInt
from omni.models.dreamllm.tokenization_dreamllm import (
    DEFAULT_DREAM_END_TOKEN,
    DEFAULT_DREAM_PATCH_TOKEN,
    DEFAULT_DREAM_START_TOKEN,
    DEFAULT_DREAM_TOKEN,
    DEFAULT_IMAGE_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from omni.utils.loguru import logger

from ..constants import DataManager
from ..manager.dataset_type import DatasetType


def truncate_and_replace(
    input_ids: list[int],
    labels: list[int],
    replacement_dict: dict[int, list[int]],
    labels_fill_value: int,
    truncate: int,
):
    new_input_ids = []
    new_labels = []
    current_position = 0

    for id, label in zip(input_ids, labels):
        if id in replacement_dict.keys():
            replacement = replacement_dict[id]
            # Check if the length after replacement will exceed the truncation point.
            if current_position + len(replacement) > truncate:
                # If truncation occurs within a replacement sequence, the entire replacement sequence is discarded
                if current_position < truncate:
                    new_input_ids = new_input_ids[: -(current_position - truncate)]
                    new_labels = new_labels[: -(current_position - truncate)]
                break  # truncate list
            new_input_ids.extend(replacement)
            new_labels.extend([labels_fill_value] * len(replacement))
            current_position += len(replacement)
        else:
            new_input_ids.append(id)
            new_labels.append(label)
            current_position += 1
            if current_position == truncate:
                break  # truncate list

    return new_input_ids, new_labels


class DreamLLMDataset(Dataset):
    def __init__(
        self,
        datasets: list[str],
        datasets_init_kwargs: dict,
        size_list: list[str | int | LargeInt],
        tokenizer: PreTrainedTokenizerBase,
        clip_vision_embedding_processor: Callable,
        stable_diffusion_head_processor: Callable,
        clip_vision_embedding_len: int,
        dream_embedding_len: int,
        comprehension_only: bool = False,
        creation_only: bool = False,
        use_sdxl_head: bool = False,
        use_image_start_and_end: bool = True,
        use_dream_start_and_end: bool = True,
        conv_template_name: str = None,
    ):
        super().__init__()
        assert not (comprehension_only and creation_only), "`comprehension_only` and `creation_only` cannot be `True` at the same time."

        self.inner_dataset = DataManager(datasets=datasets, datasets_init_kwargs=datasets_init_kwargs, size_list=size_list)
        self.tokenizer = tokenizer
        self.clip_vision_embedding_processor = clip_vision_embedding_processor
        self.stable_diffusion_head_processor = stable_diffusion_head_processor
        self.clip_vision_embedding_len = clip_vision_embedding_len
        self.dream_embedding_len = dream_embedding_len
        self.comprehension_only = comprehension_only
        self.creation_only = creation_only
        self.use_sdxl_head = use_sdxl_head
        self.use_image_start_and_end = use_image_start_and_end
        self.use_dream_start_and_end = use_dream_start_and_end
        self.conv_template = get_conv_template(conv_template_name) if conv_template_name is not None else None

    def __len__(self):
        return len(self.inner_dataset)

    def _merge_text_list(self, text_list, matched_text_index):
        new_text_list = []
        prev_index = 0
        for index in matched_text_index:
            new_text_list.append(" ".join(text_list[prev_index : index + 1]))
            prev_index = index + 1
        if prev_index != len(text_list):
            new_text_list.append(" ".join(text_list[prev_index:]))
        return new_text_list

    def _image_ids(self):
        image_patch_id, image_start_id, image_end_id = tuple(
            self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN])
        )
        image_ids = [image_patch_id] * self.clip_vision_embedding_len
        if self.use_image_start_and_end:
            image_ids = [image_start_id] + image_ids + [image_end_id]
        return image_ids

    def _dream_ids(self):
        dream_patch_id, dream_start_id, dream_end_id = tuple(
            self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_DREAM_START_TOKEN, DEFAULT_DREAM_END_TOKEN])
        )
        if self.use_sdxl_head:
            dream_patch_id = self.tokenizer.convert([DEFAULT_DREAM_PATCH_TOKEN])
        dream_ids = [dream_patch_id] * self.dream_embedding_len
        if self.use_dream_start_and_end:
            dream_ids = [dream_start_id] + dream_ids + [dream_end_id]
        return dream_ids

    def __getitem__(self, index):
        sample = self.inner_dataset.__getitem__(index)

        old_add_eos_token = self.tokenizer.add_eos_token
        self.tokenizer.add_eos_token = False

        image_id, image_patch_id, image_start_id, image_end_id = tuple(
            self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN])
        )
        dream_id, dream_patch_id, dream_start_id, dream_end_id = tuple(
            self.tokenizer.convert_tokens_to_ids([DEFAULT_DREAM_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_DREAM_START_TOKEN, DEFAULT_DREAM_END_TOKEN])
        )
        if self.use_sdxl_head:
            dream_patch_id = self.tokenizer.convert([DEFAULT_DREAM_PATCH_TOKEN])

        if sample.dataset_type == DatasetType.ImageImagePair:
            image = sample.image_source
            image_dm = sample.image_target
            image_ids = self._image_ids()
            dream_ids = self._dream_ids()
            input_ids = [self.tokenizer.bos_token_id] + image_ids + dream_ids + [self.tokenizer.eos_token_id]
            images_dm = [self.stable_diffusion_head_processor(image_dm)]
            add_time_ids = None
            try:
                images = [self.clip_vision_embedding_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]]
            except:
                images = []
                logger.warning(f"{image}")
                logger.warning("something gose wrong with images processor, skipped")
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(input_ids)

        elif sample.dataset_type == DatasetType.ImageTextTokenPair:
            image = sample.image
            text = sample.text
            input_ids = sample.info["input_ids"]

            # NOTE The samples are pre-tokenized with vicuna. Note that the special tokens are not the same as DreamLLM!
            rewrite_tokens_map = {32000: image_patch_id, 32001: image_start_id, 32002: image_end_id}
            input_ids = [_id if _id not in rewrite_tokens_map.keys() else rewrite_tokens_map[_id] for _id in input_ids]

            add_time_ids = None
            if self.comprehension_only or (not self.creation_only and not self.comprehension_only):
                attention_mask = [1] * len(input_ids)
                labels = copy.deepcopy(input_ids)
                labels = [labels[i] if sample.info["labels"][i] else IGNORE_INDEX for i in range(len(labels))]
                try:
                    images = [self.clip_vision_embedding_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]]
                except:
                    images = []
                    logger.warning(f"{image}")
                    logger.warning("something gose wrong with images processor, skipped")
                images_dm = []

        elif sample.dataset_type == DatasetType.ImageTextPair:
            image = sample.image
            text = sample.text
            # The logic of (not self.creation_only and not self.comprehension_only) in ImageTextPair is
            # randomly selecting one of the two training task, which are t2i and i2t.
            prob = np.random.rand()
            if self.comprehension_only or (not self.creation_only and not self.comprehension_only and prob >= 0.5):
                image_ids = self._image_ids()
                input_ids = self.tokenizer(
                    text,
                    max_length=self.tokenizer.model_max_length - len(image_ids) - 1,  # -1 for eos
                    truncation=True,
                ).input_ids
                input_ids = input_ids[:1] + image_ids + input_ids[1:] + [self.tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                labels = copy.deepcopy(input_ids)
                labels = [IGNORE_INDEX if x == image_patch_id or x == image_start_id or x == image_end_id else x for x in labels]
                try:
                    images = [self.clip_vision_embedding_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]]
                except:
                    images = []
                    logger.warning(f"{image}")
                    logger.warning("something gose wrong with images processor, skipped")
                images_dm = []
                add_time_ids = None
            elif self.creation_only or (not self.creation_only and not self.comprehension_only and prob < 0.5):
                dream_ids = self._dream_ids()
                input_ids = self.tokenizer(
                    text,
                    max_length=self.tokenizer.model_max_length - len(dream_ids) - 1,  # -1 for eos
                    truncation=True,
                ).input_ids
                input_ids = input_ids + dream_ids + [self.tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                labels = [IGNORE_INDEX] * len(input_ids)  # ignore all language modeling loss
                images = []
                if self.use_sdxl_head:
                    images_dm, add_time_ids = self.stable_diffusion_head_processor(image)
                    images_dm = [images_dm]
                else:
                    images_dm = [self.stable_diffusion_head_processor(image)]
                    add_time_ids = None
            else:
                raise ValueError("Should not reach here.")

            if len(input_ids) > self.tokenizer.model_max_length:
                logger.warning(f"Input length {len(input_ids)} exceeds the model max length {self.tokenizer.model_max_length}")

        elif sample.dataset_type == DatasetType.InterleavedImageText:
            image_list = sample.image_list
            text_list = sample.text_list
            matched_text_index = sample.matched_text_index
            matched_sim = sample.matched_sim

            # if matched_sim is not None:
            #     if len(matched_sim) != 0:
            #         image_list = [image_list[i] for i, sim in enumerate(matched_sim) if sim >= 0.25]
            #         matched_text_index = [matched_text_index[i] for i, sim in enumerate(matched_sim) if sim >= 0.25]

            text_list = [text.strip() for text in text_list]
            text_list = self._merge_text_list(text_list, matched_text_index)

            model_max_length = self.tokenizer.model_max_length
            add_time_ids = None
            input_ids, images, images_dm = [], [], []
            for idx, text in enumerate(text_list):
                # process text
                results = self.tokenizer(text)
                cur_input_ids_no_bos_eos = results.input_ids[1:]
                if len(input_ids) + len(cur_input_ids_no_bos_eos) + 2 > model_max_length:
                    break
                input_ids = input_ids + cur_input_ids_no_bos_eos

                # process image and drop images that are corrupted
                if idx < len(image_list):
                    if self.comprehension_only:
                        append_ids = self._image_ids()
                    elif self.creation_only:
                        append_ids = self._dream_ids()
                    else:
                        append_ids = self._dream_ids() + self._image_ids()
                    if len(input_ids) + len(append_ids) + 2 > model_max_length:
                        break

                    try:
                        if self.comprehension_only:
                            images.append(self.clip_vision_embedding_processor.preprocess(image_list[idx], return_tensors="pt")["pixel_values"][0])
                        elif self.creation_only:
                            images_dm.append(self.stable_diffusion_head_processor(image_list[idx]))
                        else:
                            images.append(self.clip_vision_embedding_processor.preprocess(image_list[idx], return_tensors="pt")["pixel_values"][0])
                            images_dm.append(self.stable_diffusion_head_processor(image_list[idx]))
                    except:
                        append_ids = []
                        logger.warning("something gose wrong with images or images_dm processor, skipped")

                    input_ids = input_ids + append_ids

            input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            labels = copy.deepcopy(input_ids)
            labels = [
                IGNORE_INDEX if x == image_patch_id or x == image_start_id or x == image_end_id or x == dream_patch_id or x == dream_end_id else x
                for x in labels
            ]  # only keep `dream_start_id` to learn

        elif sample.dataset_type == DatasetType.InstructInterleavedImageText:
            image_list = sample.image_list
            text_list = sample.text_list
            matched_text_index = sample.matched_text_index
            matched_sim = sample.matched_sim
            instruction = sample.instruction

            assert self.conv_template is not None, "The `conv_template` should be specified."
            self.conv_template.reset_dialog(sample.dialog)
            prompt = self.conv_template.get_prompt()
            input_ids_instruction = self.tokenizer(prompt).input_ids
            assert self.conv_template.sep_style == SeparatorStyle.ADD_COLON_TWO, "Only support `ADD_COLON_TWO` now."

            # if matched_sim is not None:
            #     if len(matched_sim) != 0:
            #         image_list = [image_list[i] for i, sim in enumerate(matched_sim) if sim >= 0.25]
            #         matched_text_index = [matched_text_index[i] for i, sim in enumerate(matched_sim) if sim >= 0.25]

            text_list = [text.strip() for text in text_list]
            text_list = self._merge_text_list(text_list, matched_text_index)

            model_max_length = self.tokenizer.model_max_length
            add_time_ids = None
            # input_ids, images, images_dm = [], [], []
            input_ids, images, images_dm = self.tokenizer(prompt).input_ids, [], []
            for idx, text in enumerate(text_list):
                # process text
                results = self.tokenizer(text)
                cur_input_ids_no_bos_eos = results.input_ids[1:]
                if len(input_ids) + len(cur_input_ids_no_bos_eos) + 2 > model_max_length:
                    break
                input_ids = input_ids + cur_input_ids_no_bos_eos

                # process image and drop images that are corrupted
                if idx < len(image_list):
                    if self.comprehension_only:
                        append_ids = self._image_ids()
                    elif self.creation_only:
                        append_ids = self._dream_ids()
                    else:
                        append_ids = self._dream_ids() + self._image_ids()
                    if len(input_ids) + len(append_ids) + 2 > model_max_length:
                        break

                    try:
                        if self.comprehension_only:
                            images.append(self.clip_vision_embedding_processor.preprocess(image_list[idx], return_tensors="pt")["pixel_values"][0])
                        elif self.creation_only:
                            images_dm.append(self.stable_diffusion_head_processor(image_list[idx]))
                        else:
                            images.append(self.clip_vision_embedding_processor.preprocess(image_list[idx], return_tensors="pt")["pixel_values"][0])
                            images_dm.append(self.stable_diffusion_head_processor(image_list[idx]))
                    except:
                        append_ids = []
                        logger.warning("something gose wrong with images or images_dm processor, skipped")

                    input_ids = input_ids + append_ids

            input_ids = input_ids + [self.tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            labels = copy.deepcopy(input_ids)
            labels = [
                IGNORE_INDEX if x == image_patch_id or x == image_start_id or x == image_end_id or x == dream_patch_id or x == dream_end_id else x
                for x in labels
            ]  # only keep `dream_start_id` to learn
            instruction_len = len(input_ids_instruction)
            labels[:instruction_len] = [IGNORE_INDEX] * instruction_len

        elif sample.dataset_type == DatasetType.Conversation:
            assert self.conv_template is not None, "The `conv_template` should be specified."
            self.conv_template.reset_dialog(sample.dialog)
            prompt = self.conv_template.get_prompt()
            input_ids = self.tokenizer(prompt).input_ids
            labels = copy.deepcopy(input_ids)
            assert self.conv_template.sep_style == SeparatorStyle.ADD_COLON_TWO, "Only support `ADD_COLON_TWO` now."

            # Mask targets. Only compute loss on the assistant outputs.
            sep = self.conv_template.sep + self.conv_template.roles[1] + ": "

            turns = prompt.split(self.conv_template.sep2)
            # start from bos, set it to ignore
            cur_len = 1
            labels[:cur_len] = [IGNORE_INDEX] * cur_len
            for i, turn in enumerate(turns):
                if turn == "":  # last turn is empty
                    break
                turn_len = len(self.tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not self.tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    # BUG: "USER" -> ['▁US', 'ER'], "</s>USER" -> ["</s>", "USER"] in new tokenizer.
                    instruction_len -= 1

                # Ignore the user instructions
                labels[cur_len : cur_len + instruction_len] = [IGNORE_INDEX] * instruction_len
                cur_len += turn_len

                if i != 0 and not self.tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            # FIXME: This is a hack to walk around the tokenizer bug.
            if cur_len != len(input_ids):
                logger.debug("A sample is going wroing with length, ignore it!")
            # assert cur_len == len(input_ids), f"cur_len: {cur_len}, len(input_ids): {len(input_ids)}"

            mm_content_list = []
            for message in self.conv_template.dialog:
                mm_content_list = mm_content_list + message.content.mm_content_list
            input_ids, labels = truncate_and_replace(
                input_ids=input_ids,
                labels=labels,
                replacement_dict={image_id: self._image_ids(), dream_id: self._dream_ids() + self._image_ids()},
                labels_fill_value=IGNORE_INDEX,
                truncate=self.tokenizer.model_max_length,
            )
            attention_mask = [1] * len(input_ids)

            images = []
            images_dm = []
            add_time_ids = None
            content_index = 0
            for id in input_ids:
                if id == image_start_id:
                    images.append(
                        self.clip_vision_embedding_processor.preprocess(
                            mm_content_list[content_index],
                            return_tensors="pt",
                        )[
                            "pixel_values"
                        ][0]
                    )
                    content_index += 1
                elif id == dream_start_id:
                    images_dm.append(self.stable_diffusion_head_processor(mm_content_list[content_index]))
                    # content_index will not increase here

        else:
            logger.error("Should not reach here.")

        images = torch.stack(images, 0) if len(images) > 0 else None
        images_dm = torch.stack(images_dm, 0) if len(images_dm) > 0 else None

        # reset tokenizer
        self.tokenizer.add_eos_token = old_add_eos_token

        return_dict = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "images": images,
            "images_dm": images_dm,
        }

        if self.use_sdxl_head:
            return_dict["add_time_ids"] = torch.tensor(add_time_ids) if add_time_ids is not None else None

        return return_dict


def batch_dict(list_dict: list[dict]) -> dict:
    _batch_dict = {}
    keys = list_dict[0].keys()
    for key in keys:
        _batch_dict[key] = [_dict[key] for _dict in list_dict]
    return _batch_dict


@dataclass
class DataCollatorForDreamLLMDataset:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        examples = batch_dict(examples)
        examples["input_ids"] = torch.nn.utils.rnn.pad_sequence(examples["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        examples["attention_mask"] = torch.nn.utils.rnn.pad_sequence(examples["attention_mask"], batch_first=True, padding_value=0)
        examples["labels"] = torch.nn.utils.rnn.pad_sequence(examples["labels"], batch_first=True, padding_value=IGNORE_INDEX)

        examples["images"] = [img for img in examples["images"] if img is not None]
        examples["images"] = torch.cat(examples["images"], 0) if len(examples["images"]) > 0 else None

        examples["images_dm"] = [img for img in examples["images_dm"] if img is not None]
        examples["images_dm"] = torch.cat(examples["images_dm"], 0) if len(examples["images_dm"]) > 0 else None

        return examples


@dataclass
class DataCollatorForDreamLLMSDXLDataset:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        examples = batch_dict(examples)
        examples["input_ids"] = torch.nn.utils.rnn.pad_sequence(examples["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        examples["attention_mask"] = torch.nn.utils.rnn.pad_sequence(examples["attention_mask"], batch_first=True, padding_value=0)
        examples["labels"] = torch.nn.utils.rnn.pad_sequence(examples["labels"], batch_first=True, padding_value=IGNORE_INDEX)

        examples["images"] = [img for img in examples["images"] if img is not None]
        examples["images"] = torch.cat(examples["images"], 0) if len(examples["images"]) > 0 else None

        examples["images_dm"] = [img for img in examples["images_dm"] if img is not None]
        examples["images_dm"] = torch.cat(examples["images_dm"], 0) if len(examples["images_dm"]) > 0 else None

        examples["add_time_ids"] = [add_time_ids for add_time_ids in examples["add_time_ids"] if add_time_ids is not None]
        examples["add_time_ids"] = torch.cat(examples["add_time_ids"], 0) if len(examples["add_time_ids"]) > 0 else None

        return examples
