import argparse
import json
import math
import os
import random
from typing import Any

import accelerate
import pytorch_fid
import torch
from accelerate import PartialState
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPVisionModel, LlamaTokenizer

from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM as AutoModelForCausalLM
from omni.utils.comm import all_gather
from omni.utils.loguru import logger
from omni.utils.profiler import FunctionProfiler


class COCODataset(Dataset):
    def __init__(self, root, ann_file):
        self.root = root

        self.coco = COCO(os.path.join(self.root, ann_file))
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]

        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_anns(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def _load_caption(self, key: int):
        anns = self._load_anns(key)
        captions = [ann["caption"] for ann in anns]
        return captions

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        path = self.coco.loadImgs(key)[0]["file_name"]
        return {"captions": self._load_caption(key), "image_path": path}


def collate_fn(examples):
    return examples


def load_model(model_name_or_path):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=True,
        padding_side="left",
    )

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
            cache_dir=None,
            reset_plugin_model_name_or_path=True,
            # torch_dtype=torch.bfloat16,
        ).cuda()
    model = torch.compile(model)
    return tokenizer, model


def dreamllm_ddp_sample_coco(
    model_name_or_path="/mnt/host0/stage2_1kiter",
    coco_root="./datasets/coco",
    ann_file="annotations/captions_val2014.json",
    n_samples=30000,
    batch_size_per_device=5,
    out_dir="./samples/dreamllm_coco30k",
    num_inference_steps=100,
    guidance_scale=2.5,
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

    dataset = COCODataset(root=coco_root, ann_file=ann_file)
    # ann = json.load(open(ann_file,"r"))
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

    # model, dataloader = accelerator.prepare(model,dataloader)

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
        for i, (image, _info, _path) in enumerate(zip(images, info, img_path)):
            filename = step * total_batch_size + accelerator.process_index * batch_size_per_device + i
            image.save(os.path.join(out_dir, f"{filename:05}.png"))
        pbar.update(1)


"""
torchrun --master_addr $MLP_WORKER_0_HOST --master_port $MLP_WORKER_0_PORT --node_rank $MLP_ROLE_INDEX --nnodes $MLP_WORKER_NUM --nproc_per_node $MLP_WORKER_GPU dreamllm/eval/text2img/ddp_sample_coco.py \
--type sd \
--diffusion_model_name_or_path stabilityai/stable-diffusion-2-1 \
--clip_model_name_or_path openai/clip-vit-large-patch14 \
--is_fp16 \
--coco_root path2data \
--ann_file data/captions_val2014.json \
--n_samples 30000 \
--batch_size_per_device 1 \
--out_dir samples/stable_coco30k \
--guidance_scale 2.0 \
--local_files_only
"""


def stable_ddp_sample_coco(
    model_name_or_path="runwayml/stable-diffusion-v1-5",
    is_fp16=False,
    local_files_only=False,
    coco_root="./datasets/coco",
    ann_file="annotations/captions_val2014.json",
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
                index = len(captions)
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
            inputs = processor(text=info["captions"], images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            index = torch.argmax(logits_per_image) // logits_per_image.shape[1]

            images[index].save(os.path.join(base_dir, out_dir, f"{key:05}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="sd", choices=["sd", "dreamllm", "caption", "select"])
    parser.add_argument("--model_name_or_path", type=str, default="/mnt/host0/stage2_1kiter")
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
        dirs_list = [f"{args.dirs_name}{i}" for i in range(42, 50)]
        select_image(
            data_info_path=args.data_info_path,
            clip_for_similarity_model_name_or_path=args.clip_for_similarity_model_name_or_path,
            base_dir=args.base_dir,
            dirs_list=dirs_list,
            out_dir=args.out_dir,
        )
