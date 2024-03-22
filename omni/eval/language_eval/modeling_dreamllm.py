import os
import time
import json
import math
import torch
import sys

from typing import List
from pathlib import Path
from types import MethodType
from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM as AutoModelForCausalLM

# from llava.model.llama import ModelArgs, Transformer, LLaMA
from transformers import LlamaTokenizer
import transformers
from safetensors import safe_open
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

DTYPE = torch.float32


def setup_model_parallel():
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return rank, world_size


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate(
    self,
    prompts: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_len: int,
    max_gen_len: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> List[str]:
    bsz = len(prompts)
    # params = self.model.params
    # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    min_prompt_size = min([len(t) for t in prompt_tokens])
    max_prompt_size = max([len(t) for t in prompt_tokens])

    # total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
    total_len = min(max_seq_len, max_gen_len + max_prompt_size)

    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id).cuda().long()
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()
    input_text_mask = tokens != tokenizer.pad_token_id
    start_pos = min_prompt_size
    prev_pos = 0
    mask = None
    for cur_pos in range(start_pos, total_len):
        if cur_pos == start_pos:
            outputs = self.forward(input_ids=tokens[:, prev_pos:cur_pos], use_cache=True)
            logits = outputs.logits[..., -1, :32000]
            past_key_values = outputs.past_key_values
        else:
            attention_mask = torch.ones(bsz, past_key_values[0][0].shape[-2] + 1, device="cuda")
            out = self.forward(
                input_ids=tokens[:, prev_pos:cur_pos], use_cache=True, attention_mask=attention_mask, output_hidden_states=True, past_key_values=past_key_values
            )
            logits = out.logits[..., -1, :32000]
            past_key_values = out.past_key_values
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos

    decoded = []
    for i, t in enumerate(tokens.tolist()):
        # cut to max gen len
        t = t[: len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        try:
            t = t[: t.index(tokenizer.eos_token_id)]
        except ValueError:
            pass
        decoded.append(tokenizer.decode(t))
    return decoded


def load(ckpt_dir: str, num_gpus: int, max_seq_len=1024, max_batch_size=32, add_special_pattern=False):
    start_time = time.time()

    print(f"loading from {ckpt_dir}")
    tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage=True, torch_dtype=torch.float32)

    model.cuda()

    # model.tokenizer = tokenizer
    model.generate = MethodType(generate, model)

    return model, tokenizer

def main(
    ckpt_dir: str,
    num_gpus: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    model, tokenizer = load(ckpt_dir, num_gpus)

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>""",
    ]
    results = model.generate(prompts, tokenizer, max_gen_len=256, max_seq_len=max_seq_len, temperature=temperature, top_p=top_p)

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    ckpt_dir = "path2model"
    num_gpus = 1
    main(ckpt_dir, num_gpus)
