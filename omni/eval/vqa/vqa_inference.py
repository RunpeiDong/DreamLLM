import base64
import json
import math
import os
from io import BytesIO

import pandas as pd
import requests
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel, LlamaTokenizer

from omni.constants import (
    DEFAULT_IMAGE_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_START_TOKEN,
    MODEL_ZOOS,
)
from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM as LlamaForCausalLM
from omni.utils.conversation import KeywordsStoppingCriteria, SeparatorStyle, default_conversation
from omni.utils.eval_utils import get_parser
from omni.utils.image_utils import load_image, save_image
from omni.utils.loguru import logger
from omni.utils.misc import disable_torch_init
from omni.utils.profiler import FunctionProfiler


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_chunk_pd(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k].iloc


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image


def load_from_df(ann, key):
    if key in ann and not pd.isna(ann[key]):
        return ann[key]
    else:
        return None


def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    image = image.resize((new_width, new_height))
    width_diff = target_size[0] - image.size[0]
    height_diff = target_size[1] - image.size[1]
    left_padding = 0
    top_padding = 0
    right_padding = width_diff - left_padding
    bottom_padding = height_diff - top_padding
    padded_image = ImageOps.expand(image, border=(left_padding, top_padding, right_padding, bottom_padding), fill=0)
    return padded_image


output_list = []


def get_img_tensor(image_file_path, image_processor):
    image = load_image(image_file_path)
    if args.img_aug == "square_resize":
        image = image.resize((224, 224))
    elif args.img_aug == "padding_square_resize":
        image = resize_image(image, (224, 224))
    elif args.img_aug == "center_crop":
        image = image.resize((256, 256))
        image = image.crop((16, 16, 240, 240))
    elif args.img_aug == "none":
        pass
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image_tensor


def generate(args, tokenizer, conv, image_tensor, model):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    prompt = conv.get_prompt()
    if prompt.endswith(stop_str):
        prompt = prompt[: -len(stop_str)]

    inputs = tokenizer([prompt])

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if str(args.beamsearch) == "True":
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                num_beams=5,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria],
            )
        elif str(args.beamsearch) == "short":
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=128,
                stopping_criteria=[stopping_criteria],
            )
        else:
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria],
            )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        logger.warning(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    return outputs


def get_caption(args, tokenizer, image_token_len, image_file_path, image_processor, model):
    prompt = "please provide an accurate and concise description of the given image." if args.prompt == "None" else args.prompt
    post_prompt = "" if args.post_prompt == "None" else args.post_prompt
    system_prompt = "" if args.system_prompt == "None" else args.system_prompt
    qs = (
        system_prompt
        + DEFAULT_IMAGE_START_TOKEN
        + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        + DEFAULT_IMAGE_END_TOKEN
        + prompt
    )

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], post_prompt)

    image_tensor = get_img_tensor(image_file_path, image_processor)
    outputs = generate(args, tokenizer, conv, image_tensor, model)
    outputs = outputs.capitalize()

    return outputs


def get_vqa_result(args, tokenizer, image_token_len, image_file_path, image_processor, model, qs):
    if args.datatype == "VizWizVQA":
        system_prompt = ""
        f_qs = DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IMAGE_END_TOKEN + qs
        f_qs += "Is the answer can be known?"
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], f_qs)
        conv.append_message(conv.roles[1], None)
        image_tensor = get_img_tensor(image_file_path, image_processor)
        outputs = generate(args, tokenizer, conv, image_tensor, model)

        if "no" in outputs.lower():
            return "unanswerable"

    # Based on the content of the image and common sense, please provide an accurate answer consisting of only one word or phrase.
    prompt = "" if args.prompt == "None" else args.prompt
    post_prompt = "" if args.post_prompt == "None" else args.post_prompt
    system_prompt = "" if args.system_prompt == "None" else args.system_prompt
    # qs = system_prompt + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IMAGE_END_TOKEN + qs + prompt
    qs = (
        system_prompt
        + DEFAULT_IMAGE_START_TOKEN
        + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        + DEFAULT_IMAGE_END_TOKEN
        + qs
        + prompt
    )
    # qs = system_prompt + DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IMAGE_END_TOKEN + "\n" + qs + prompt

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], post_prompt)

    image_tensor = get_img_tensor(image_file_path, image_processor)
    outputs = generate(args, tokenizer, conv, image_tensor, model)
    if args.clip == "True":
        outputs = outputs.replace(".", "").replace('"', "")
        outputs = outputs.split(",")[0]
        outputs = outputs.lower()

    return outputs


def get_mm_result(args, tokenizer, image_token_len, image_processor, model, ann):
    qs = ann["question"]
    encode_image = ann["image"]
    image = decode_base64_to_image(encode_image)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    qs = "Hint:" + str(ann.get("hint", "")) + "Question:" + ann["question"]

    system_prompt = ""
    f_qs = DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IMAGE_END_TOKEN + qs
    f_qs += "Is the answer can be known?"
    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], f_qs)
    conv.append_message(conv.roles[1], None)
    outputs = generate(args, tokenizer, conv, image_tensor, model)

    if "no" == outputs.lower()[:2]:
        ###
        sys_prompt = "Options:"
        # sys_prompt = 'There are several options:'
        option_candidate = ["A", "B", "C", "D", "E"]
        options = {cand: load_from_df(ann, cand) for cand in option_candidate if load_from_df(ann, cand) is not None}
        options_prompt = f"{sys_prompt} "
        i = 0
        for key, item in options.items():
            # options_prompt += f'{key}. {item} '
            options_prompt += f"({str(i)}). {item} "

        qs = qs + options_prompt

    # Based on the content of the image and common sense, please provide an accurate answer consisting of only one word or phrase.
    prompt = "" if args.prompt == "None" else args.prompt
    post_prompt = "" if args.post_prompt == "None" else args.post_prompt
    system_prompt = "" if args.system_prompt == "None" else args.system_prompt
    qs = (
        system_prompt
        + DEFAULT_IMAGE_START_TOKEN
        + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        + DEFAULT_IMAGE_END_TOKEN
        + qs
        + prompt
    )

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], post_prompt)

    outputs = generate(args, tokenizer, conv, image_tensor, model)

    return outputs


def main(args):
    disable_torch_init()
    model_name_or_path = os.path.expanduser(args.model_name)
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=True,
        padding_side="left",
        use_fast=False,
    )
    config = DreamLLMConfig.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        local_files_only=True,
        model_max_length=2048,
        use_fast=False,
    )
    with FunctionProfiler("DreamLLMForCausalMLM.from_pretrained"):
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            tokenizer=tokenizer,
            config=config,
            local_files_only=True,
            cache_dir=None,
            torch_dtype=torch.bfloat16,
            reset_plugin_model_name_or_path=True,
        ).cuda()
    model.eval()
    model = torch.compile(model)

    path = MODEL_ZOOS["openai/clip-vit-large-patch14"]
    vision_tower = CLIPVisionModel.from_pretrained(path, local_files_only=True)
    image_processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    gts_path = args.gtfile_path
    if args.task_type == "mm":
        gts = pd.read_csv(gts_path, sep="\t")
        gts = get_chunk_pd(gts, args.num_chunks, args.chunk_idx)
    else:
        gts = json.load(open(gts_path, encoding="utf-8"))
        gts = get_chunk(gts, args.num_chunks, args.chunk_idx)

    for ann in tqdm(gts):
        output_json = {}

        qs_id = ann.get("question_id", "")
        qs = ann.get("question", "")

        image_file = ann.get("image", "")
        image_file_path = os.path.join(args.image_path, image_file)

        if args.task_type == "caption":
            outputs = get_caption(args, tokenizer, image_token_len, image_file_path, image_processor, model)
            logger.info("Question: {} \nPrediction: {} \nGT: {}".format(qs, outputs, ann.get("answer", None)))
            output_json["question_id"] = qs_id
            output_json["answer"] = outputs
            if "answer" in ann:
                output_json["gt"] = ann["answer"]

        if args.task_type == "vqa" or args.task_type == "POPE":
            outputs = get_vqa_result(args, tokenizer, image_token_len, image_file_path, image_processor, model, qs)
            logger.info("Question: {} \nPrediction: {} \nGT: {}".format(qs, outputs, ann.get("answer", None)))
            output_json["question_id"] = int(qs_id)
            output_json["answer"] = outputs
            output_json["image"] = image_file
            if "answer" in ann:
                output_json["gt"] = ann["answer"]

        if args.task_type == "mm":
            outputs = get_mm_result(args, tokenizer, image_token_len, image_processor, model, ann)
            logger.info("Question: {} \nPrediction: {} \nGT: {}".format(ann["question"], outputs, ann.get("answer", None)))
            output_json["question"] = ann["question"]
            output_json["A"] = ann["A"]
            output_json["B"] = ann["B"]
            output_json["C"] = ann["C"]
            output_json["D"] = ann["D"]
            output_json["prediction"] = outputs
            output_json["category"] = ann["category"]
            output_json["l2-category"] = ann["l2-category"]
            output_json["index"] = int(ann["index"])
            output_json["answer"] = ann["answer"]

        output_list.append(output_json)

    filename = args.out_path + "/results_" + str(args.chunk_idx) + ".json"
    with open(filename, "w", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(output_list))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(args)

    main(args)
