#!/usr/bin/env python3

import argparse
import os
import re
import requests
import time
from omni.utils.loguru import logger
from typing import List, Dict
from llama.tokenizer import Tokenizer

from .modeling_llama import load, load_orin

__all__ = [
    "online_forward",
    "add_common_args",
    "extract_text",
    "load_model_by_args",
    "modelname_by_path",
    "seed_everything",
    "sft_prompt",
    "sft_answer",
    "judge",
    "tokenize",
    "monkey_patch_llama",
]


def post_and_retry(url, json, headers, retry=3):
    """Auto handle the response and return a post request response"""
    proxies = {"http": [], "https": []}
    response = requests.post(url, json=json, headers=headers, proxies=proxies)
    while response.status_code != 200 and retry > 0:
        if response.status_code == 422:
            try:
                error_msg = response.json()["error"]
            except Exception:
                error_msg = response.content.decode("utf-8")
            logger.warning(f"422 error due to {error_msg}, Plz double check your input:\n{json}.")
            break
        else:
            logger.info(f"retrying... curl {url}")
            time.sleep(3)
            response = requests.post(url, json=json, headers=headers, proxies=proxies)
        retry -= 1

    return response


def online_forward(
    input_str: str, ip_port: str, max_gen_length: int = 512,
    temperature=0.1, top_p=None, top_k=None,
    do_sample=True, details=False, seed=None,
) -> Dict:
    """Generate text using TGI online service."""
    url = "http://{}/generate".format(ip_port)
    data = {
        "inputs": input_str,
        "parameters": {
            "max_new_tokens": max_gen_length,
            "do_sample": do_sample,
            "temperature": temperature,
            "details": details,
            "seed": seed,
        }
    }
    if top_k is not None:
        data["parameters"]["top_k"] = top_k
    if top_p is not None:
        data["parameters"]["top_p"] = top_p

    headers = {
        "Content-Type": "application/json"
    }
    response = post_and_retry(url, data, headers)
    response = response.json()
    return response


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", "-m", type=str, default=None, help="Model name to database.")
    parser.add_argument("--model_size", type=str, default="7b", help="Model size to load parameters.")
    parser.add_argument(
        "--load_orin", action="store_true",
        help="load origin torch mode checkpoints, default load safetensors mode"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="path2model",
        help="Path to the checkpoint directory."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="path2model",
        help="Path to the tokenizer."
    )
    parser.add_argument(
        "--addr", type=str, default=None,
        help="Address to access TGI"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=2048,
        help="Max sequence length of LLM. Default: 2048",
    )
    parser.add_argument(
        "--topk", type=int, default=None,
        help="topk value, should be positive, defalut: None"
    )
    parser.add_argument(
        "--topp", type=float, default=None,
        help="topp value, range [0, 1], defalut: None"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="model temperature value, defalut: 0.1"
    )
    parser.add_argument(
        "--sft", action="store_true",
        help="benchmark sft model, defalut not sft"
    )
    parser.add_argument(
        "--no_db", action="store_true",
        help="not using database to store evaluate result, defalut writing to database"
    )
    parser.add_argument(
        "--logfile", type=str, default=None,
        help="log file of inference result. Default is None, which means generate by script.",
    )
    return parser


def sft_prompt(prompt: str, prefix: str = "Human: {}\n\nAssistant:") -> str:
    return prefix.format(prompt)


def sft_answer(text: str) -> str:
    return text.split("\n\nAssistant:")[-1].strip()


def load_model_by_args(args, local_rank, world_size):

    load_func = load_orin if args.load_orin else load
    # NOTE: The following logic is a bit hacky, fix it later
    kwargs = {}
    if args.ckpt_dir.startswith("data/weights") and not args.load_orin:  # load from weights with safetensors
        kwargs["vocab_align_size"] = 32
        load_func = load

    model = load_func(
        args.ckpt_dir, args.tokenizer_path, local_rank, world_size,
        args.max_seq_len, args.batch_size, args.model_size, **kwargs
    )
    return model


def modelname_by_path(ckpt_path: str) -> str:
    # path example: llama_65b
    path = ckpt_path.split(os.sep)
    keys = ["checkpoints", "weights"]
    suffix = None
    for k in keys:
        if k in path:
            suffix = path[path.index(k) + 1:]
            break
    assert suffix is not None, f"ckpt_path should contain value in {keys}"
    return "_".join(suffix[:2])


def seed_everything(seed: int) -> int:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    return seed


def extract_text(prompts: str) -> List[str]:
    # example of text showing below
    # Human: content\n\nAssistant: content\nHuman: content\n\nAssistant:   # noqa
    return re.findall(r"Human:.*?\n\nAssistant:.*?(?=(?:Human:)|$)", prompts, re.DOTALL)


def dataset_mapping(tasks):
    def _replace_tasks(task, subjects):
        if task in tasks:
            tasks.remove(task)
            tasks.extend(["/".join(task, i) for i in subjects])
    return tasks


def judge(_output: str = None, judge_code: str = None) -> float:
    x = _output
    out = eval(judge_code, locals())
    out = float(out)
    out = min(max(out, 0), 1)
    return out


def tokenize(text, tokenizer: Tokenizer, bos: int = 1, eos: int = 2, new_line: int = 13) -> List[int]:
    # NOTE: The last token does not append the eos and new_line
    sep_text = extract_text(text)
    tokens = [bos]
    for t in sep_text[:-1]:
        encode_token = tokenizer.encode(t.strip(), bos=False, eos=False) + [eos, new_line, new_line]
        tokens.extend(encode_token)
    tokens.extend(tokenizer.encode(sep_text[-1], bos=False, eos=False))
    return tokens


def monkey_patch_llama():
    import torch
    from llama.generation import sample_top_p
    from llama import LLaMA

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        apply_prompts_as_token: bool = False,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        if apply_prompts_as_token:
            prompt_tokens = prompts
            assert isinstance(prompt_tokens, List)
        else:
            prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        per_token_len = [len(t) for t in prompt_tokens]
        min_prompt_size = min(per_token_len)
        max_prompt_size = max(per_token_len)

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for token_len, t in zip(per_token_len, tokens.tolist()):
            # cut to max gen len
            t = t[: token_len + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id, token_len)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    LLaMA.generate = generate
    return LLaMA
