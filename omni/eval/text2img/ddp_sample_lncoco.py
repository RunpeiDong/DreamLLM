import argparse
import json
import math
import os
import random
from typing import Any

import accelerate
import jsonlines
import torch
from accelerate import PartialState
from accelerate.utils import set_seed
from omni.utils.profiler import FunctionProfiler

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from omni.utils.loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPVisionModel, LlamaTokenizer

from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM as AutoModelForCausalLM
from omni.constants import *
from omni.utils.comm import all_gather
from omni.utils.image_utils_deprecated import save_image

"""
accelerate launch --num_processes 8 dreamllm/eval/text2img/ddp_sample_lncoco.py \
--type sd \
--diffusion_model_name_or_path model2path \
--clip_model_name_or_path openai/clip-vit-large-patch14 \
--coco_root ./data/coco_fid_files/ \
--ann_file lncoco_captions_val2017.jsonl \
--n_samples 30000 \
--batch_size_per_device 32 \
--out_dir samples/ln_coco/seed42 \
--num_inference_steps 100 \
--guidance_scale 3.0 \
--seed 42 \
--is_fp16
"""


class COCODataset(Dataset):
    def __init__(self, root, ann_file):
        self.root = root
        self.ann_file = ann_file

        self.image_info = {}
        self.index2path = []
        with jsonlines.open(os.path.join(self.root, self.ann_file)) as reader:
            for obj in reader:
                image_path = f"{int(obj['image_id']):012d}.jpg"
                if image_path not in self.image_info.keys():
                    self.image_info[image_path] = [obj["caption"]]
                    self.index2path.append(image_path)
                else:
                    self.image_info[image_path].append(obj["caption"])

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        index = index % len(self.image_info)
        image_path = self.index2path[index]
        captions = self.image_info[image_path]
        return {"captions": captions, "image_path": image_path}


def collate_fn(examples):
    lengths = [len(example["captions"]) for example in examples]
    max_length = max(lengths)

    for example in examples:
        if len(example["captions"]) != max_length:
            example["captions"] = example["captions"] + [""] * (max_length - len(example))

    captions = [example["captions"] for example in examples]
    image_path = [example["image_path"] for example in examples]

    batch = {"captions": captions, "image_path": image_path}
    return examples


def load_model(model_name_or_path):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=True,
        padding_side="left",
    )

    with torch.device("cuda"):
        config = DreamLLMConfig.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        with FunctionProfiler("AutoModelForCausalLM.from_pretrained"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                tokenizer=tokenizer,
                config=config,
                local_files_only=True,
            )
    return tokenizer, model


def dreamllm_ddp_sample_coco(
    model_name_or_path="/mnt/host0/stage2_1kiter",
    coco_root="./datasets/coco",
    ann_file="annotations/captions_train2014.json",
    n_samples=30000,
    batch_size_per_device=5,
    out_dir="samples/dreamllm_coco30k",
    num_inference_steps=100,
    guidance_scale=5.0,
    seed=42,
):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    accelerator = accelerate.Accelerator()

    os.makedirs(out_dir, exist_ok=True)

    set_seed(seed, device_specific=True)

    device = accelerator.device
    logger.info(f"Process {accelerator.process_index} using device: {device}")

    tokenizer, model = load_model(model_name_or_path)
    model.stable_diffusion_head.set_progress_bar_config(disable=True)

    dataset = COCODataset(root=coco_root, ann_file=ann_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_device,
        collate_fn=lambda examples: collate_fn(examples),
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # model, dataloader = accelerator.prepare(model, dataloader)
    model.to(accelerator.device)
    total_batch_size = batch_size_per_device * accelerator.num_processes

    def get_caption_generator():
        while True:
            for data in dataloader:
                yield data

    caption_generator = get_caption_generator()

    steps = math.ceil(n_samples / total_batch_size)

    if accelerator.is_main_process:
        logger.info("***** Running sampling *****")
        logger.info(f"  Batch size per device = {batch_size_per_device}")
        logger.info(f"  Num processes = {accelerator.num_processes}")
        logger.info(f"  Total batch size = {total_batch_size}")
        logger.info(f"  Num steps = {steps}")
        logger.info(f"  Num samples = {steps * total_batch_size}")

    pbar = tqdm(range(steps), total=steps, disable=not accelerator.is_main_process)

    for step in range(steps):
        info = next(caption_generator)
        prompt = []
        images = []
        img_path = []
        for _info in info:
            captions = _info["captions"]
            _path = _info["image_path"]
            index = len(captions)
            for i, caption in enumerate(captions):
                if caption == "":
                    index = i
                    break

            prompt.append(random.choice(captions[:index]))
            img_path.append(_path)

        images.extend(
            model.stable_diffusion_pipeline(
                tokenizer=tokenizer,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
        )

        for i, (image, _info) in enumerate(zip(images, info)):
            filename = step * total_batch_size + accelerator.process_index * batch_size_per_device + i
            image.save(os.path.join(out_dir, f"{filename:05}.png"))

        pbar.update(1)


def stable_ddp_sample_coco(
    model_name_or_path="runwayml/stable-diffusion-v1-5",
    is_fp16=False,
    local_files_only=False,
    coco_root="./datasets/coco",
    ann_file="annotations/captions_train2014.json",
    n_samples=30000,
    batch_size_per_device=30,
    out_dir="samples/stable_coco30k",
    num_inference_steps=50,
    guidance_scale=5.0,
    ddim_eta=0.0,
    seed=42,
):
    os.makedirs(out_dir, exist_ok=True)

    torch_dtype = torch.float32
    if is_fp16:
        torch_dtype = torch.float16

    scheduler = DDIMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler", local_files_only=local_files_only)
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name_or_path, scheduler=scheduler, torch_dtype=torch_dtype, local_files_only=local_files_only
    )
    pipeline.set_progress_bar_config(disable=True)

    distributed_state = PartialState()
    pipeline.to(distributed_state.device)
    logger.info(f"Process {distributed_state.process_index} using device: {distributed_state.device}")

    dataset = COCODataset(root=coco_root, ann_file=ann_file)

    total_batch_size = batch_size_per_device * distributed_state.num_processes
    dataloader = DataLoader(
        dataset,
        batch_size=total_batch_size,
        collate_fn=lambda examples: collate_fn(examples),
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    def get_caption_generator():
        while True:
            for data in dataloader:
                yield data

    caption_generator = get_caption_generator()

    steps = math.ceil(n_samples / total_batch_size)

    if distributed_state.is_main_process:
        logger.info("***** Running sampling *****")
        logger.info(f"  Batch size per device = {batch_size_per_device}")
        logger.info(f"  Num processes = {distributed_state.num_processes}")
        logger.info(f"  Total batch size = {total_batch_size}")
        logger.info(f"  Num steps = {steps}")
        logger.info(f"  Num samples = {steps * total_batch_size}")

    pbar = tqdm(range(steps), total=steps, disable=not distributed_state.is_main_process)

    for step in range(steps):
        with distributed_state.split_between_processes(next(caption_generator)) as info:
            prompt = []
            for _info in info:
                captions = _info["captions"]
                index = -1
                for i, caption in enumerate(captions):
                    if caption == "":
                        index = i
                        break
                prompt.append(random.choice(captions[:index]))

            images = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=ddim_eta,
                generator=torch.Generator().manual_seed(seed),
            ).images

            for i, image in enumerate(images):
                filename = step * total_batch_size + distributed_state.process_index * batch_size_per_device + i
                image.save(os.path.join(out_dir, f"{filename:05}.png"))

        pbar.update(1)


def caption_info(
    coco_root="./datasets/coco",
    ann_file="annotations/captions_train2014.json",
    n_samples=30000,
    batch_size_per_device=5,
    out_data_info_path="samples/all_caption/data_info.json",
):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    accelerator = accelerate.Accelerator()

    device = accelerator.device
    logger.info(f"Process {accelerator.process_index} using device: {device}")

    dataset = COCODataset(root=coco_root, ann_file=ann_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_device,
        collate_fn=lambda examples: collate_fn(examples),
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    dataloader = accelerator.prepare(dataloader)

    total_batch_size = batch_size_per_device * accelerator.num_processes

    def get_caption_generator():
        while True:
            for data in dataloader:
                yield data

    caption_generator = get_caption_generator()

    steps = math.ceil(n_samples / total_batch_size)

    if accelerator.is_main_process:
        logger.info("***** Running sampling *****")
        logger.info(f"  Batch size per device = {batch_size_per_device}")
        logger.info(f"  Num processes = {accelerator.num_processes}")
        logger.info(f"  Total batch size = {total_batch_size}")
        logger.info(f"  Num steps = {steps}")
        logger.info(f"  Num samples = {steps * total_batch_size}")

    pbar = tqdm(range(steps), total=steps, disable=not accelerator.is_main_process)

    data_info = {}
    for step in range(steps):
        info = next(caption_generator)

        for i, _info in enumerate(info):
            filename = step * total_batch_size + accelerator.process_index * batch_size_per_device + i
            data_info[f"{filename:05}"] = _info

        pbar.update(1)

    data_list = all_gather(data_info, verbose=True)

    merge_dict = {}
    if accelerator.is_main_process:
        for data in data_list:
            merge_dict.update(data)

        os.makedirs(os.path.dirname(out_data_info_path), exist_ok=True)
        with open(out_data_info_path, "w") as outfile:
            json.dump(merge_dict, outfile)

        logger.info("Finished dumpping data infomation ...")


def select_image(
    data_info_path: str = "samples/all_caption/data_info.json",
    clip_for_similarity_model_name_or_path: str = "openai/clip-vit-large-patch14",
    local_files_only: str = False,
    base_dir: str = "samples",
    dirs_list: list = ["output1", "output2", "output3", "output4", "output5", "output6", "output7", "output8"],
    out_dir: str = "output_res",
):
    accelerator = accelerate.Accelerator()

    with open(data_info_path, "r") as f:
        data_info = json.load(f)

    model = CLIPModel.from_pretrained(clip_for_similarity_model_name_or_path, local_files_only=local_files_only)
    processor = CLIPProcessor.from_pretrained(clip_for_similarity_model_name_or_path, local_files_only=local_files_only)

    model.to(accelerator.device)

    step = len(data_info) // accelerator.num_processes

    tuple_data_info = tuple(data_info.items())

    for i in tqdm(range(step), disable=not accelerator.is_main_process):
        with accelerator.split_between_processes(
            tuple_data_info[accelerator.num_processes * i : accelerator.num_processes * i + accelerator.num_processes]
        ) as _data_info:
            key, info = _data_info[0]
            images = [Image.open(os.path.join(base_dir, dir, f"{key}.png")).convert("RGB") for dir in dirs_list]
            inputs = processor(text=info["captions"], images=images, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            index = torch.argmax(logits_per_image) // logits_per_image.shape[1]

            save_image(images[index], os.path.join(base_dir, out_dir, f"{key:05}.png"))
            # images[index].save(os.path.join(base_dir, out_dir, f"{key:05}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="sd", choices=["sd", "dreamllm", "caption", "select"])
    parser.add_argument("--model_name_or_path", type=str, default="path2model")
    parser.add_argument("--diffusion_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--clip_model_name_or_path", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_for_similarity_model_name_or_path", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--coco_root", type=str, default="./datasets/coco")
    parser.add_argument("--ann_file", type=str, default="annotations/captions_val2014.json")
    parser.add_argument("--n_samples", type=int, default=30000)
    parser.add_argument("--batch_size_per_device", type=int, default=30)
    parser.add_argument("--base_dir", type=str, default="samples/stable_coco30k")
    parser.add_argument("--out_dir", type=str, default="samples/stable_coco30k")
    parser.add_argument("--dirs_name", type=str, default="seed")
    parser.add_argument("--data_info_path", type=str, default="samples/stable_coco30k/data_info.json")
    parser.add_argument("--out_data_info_path", type=str, default="samples/stable_coco30k/data_info.json")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.type == "sd":
        stable_ddp_sample_coco(
            model_name_or_path=args.diffusion_model_name_or_path,
            is_fp16=args.is_fp16,
            local_files_only=args.local_files_only,
            coco_root=args.coco_root,
            ann_file=args.ann_file,
            n_samples=args.n_samples,
            batch_size_per_device=args.batch_size_per_device,
            out_dir=args.out_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ddim_eta=args.ddim_eta,
            seed=args.seed,
        )
    elif args.type == "dreamllm":
        dreamllm_ddp_sample_coco(
            model_name_or_path=args.model_name_or_path,
            coco_root=args.coco_root,
            ann_file=args.ann_file,
            n_samples=args.n_samples,
            batch_size_per_device=args.batch_size_per_device,
            out_dir=args.out_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    elif args.type == "caption":
        caption_info(
            coco_root=args.coco_root,
            ann_file=args.ann_file,
            n_samples=args.n_samples,
            batch_size_per_device=args.batch_size_per_device,
            out_data_info_path=args.out_data_info_path,
        )
    elif args.type == "select":
        dirs_list = [f"{args.dirs_name}{i}" for i in range(42, 58)]
        select_image(
            data_info_path=args.data_info_path,
            clip_for_similarity_model_name_or_path=args.clip_for_similarity_model_name_or_path,
            base_dir=args.base_dir,
            dirs_list=dirs_list,
            out_dir=args.out_dir,
        )
