#!/usr/bin/env python3

import re
import os
import argparse
from typing import List

# from llama.tokenizer import Tokenizer
from transformers import LlamaTokenizer

__all__ = [
    "add_common_args",
    "extract_text",
    "load_model_by_args",
    "modelname_by_path",
    "seed_everything",
    "sft_prompt",
    "sft_answer",
    "tokenize",
]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", "-m", type=str, default=None, help="Model name to database.")
    parser.add_argument("--model_size", type=str, default="7b", help="Model size to load parameters.")
    parser.add_argument("--load_orin", action="store_true", help="load origin torch mode checkpoints, default load safetensors mode")
    parser.add_argument("--ckpt_dir", type=str, default="path2model", help="Path to the checkpoint directory.")
    parser.add_argument("--tokenizer_path", type=str, default="path2model", help="Path to the tokenizer.")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Max sequence length of LLM. Default: 2048",
    )
    parser.add_argument("--topk", type=int, default=-1, help="topk value, should be positive, defalut: -1")
    parser.add_argument("--topp", type=float, default=0.95, help="topp value, range [0, 1], defalut: 0.95")
    parser.add_argument("--temperature", type=float, default=0.1, help="model temperature value, defalut: 0.1")
    parser.add_argument("--sft", action="store_true", help="benchmark sft model, defalut not sft")
    parser.add_argument("--no_db", action="store_true", help="not using database to store evaluate result, defalut writing to database")
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="log file of inference result. Default is None, which means generate by script.",
    )
    return parser


def sft_prompt(prompt: str, prefix: str = "Human: {}\n\nAssistant: ") -> str:
    return prefix.format(prompt)


def sft_answer(text: str) -> str:
    return text.split("\n\nAssistant: ")[-1].strip()


def load_model_by_args(args, local_rank, world_size):
    from omni.eval.language_eval.modeling_dreamllm import load, load_orin

    load_func = load_orin if args.load_orin else load
    model = load_func(args.ckpt_dir, args.tokenizer_path, local_rank, world_size, args.max_seq_len, args.batch_size, args.model_size)
    return model


def modelname_by_path(ckpt_path: str) -> str:
    # path example: llama_65b
    path = ckpt_path.split(os.sep)
    keys = ["checkpoints", "weights", "work_dirs"]
    suffix = None
    for k in keys:
        if k in path:
            suffix = path[path.index(k) + 1 :]
            break
    assert suffix is not None, f"ckpt_path should contain value in {keys}"
    return "_".join(suffix[:2])


def seed_everything(seed: int) -> int:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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


def tokenize(text, tokenizer: LlamaTokenizer, bos: int = 1, eos: int = 2, new_line: int = 13) -> List[int]:
    # NOTE: The last token does not append the eos and new_line
    sep_text = extract_text(text)
    tokens = [bos]
    for t in sep_text[:-1]:
        encode_token = tokenizer.encode(t.strip(), bos=False, eos=False) + [eos, new_line, new_line]
        tokens.extend(encode_token)
    tokens.extend(tokenizer.encode(sep_text[-1], bos=False, eos=False))
    return tokens
