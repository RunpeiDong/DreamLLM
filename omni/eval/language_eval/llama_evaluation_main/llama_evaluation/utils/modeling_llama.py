import os
import time
import json
import math
import torch
import sys

from pathlib import Path
from types import MethodType
from llama import ModelArgs, Transformer, Llama
from llama.tokenizer import Tokenizer
from safetensors import safe_open
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

__all__ = [
    "MODEL_PARAMS", "setup_model_parallel", "forward_all", "load", "load_orin",
]

MODEL_PARAMS = {
    "1_5b": {"dim":2048, "multiple_of": 256, "n_heads": 32, "n_layers": 24, "norm_eps": 1e-5, "vocab_size": 65536},
    "1b": {"dim": 2048, "multiple_of": 256, "n_heads": 16, "n_layers": 16, "norm_eps": 1e-5, "vocab_size": 65536},
    "7b": {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-5, "vocab_size": 55808},
    "65b": {"dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-5, "vocab_size": 55808},
}


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


@torch.inference_mode()
def forward_all(self, tokens: torch.Tensor, start_pos: int):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

    mask = None
    if seqlen > 1:
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

    for layer in self.layers:
        h = layer(h, start_pos, freqs_cis, mask)
    h = self.norm(h)

    return self.output(h).float() # compute all logits


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    model_size: str,
    vocab_align_size: int = 512,
):
    # vocab_align_size = 32 for origin LLaMA 7B
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
    ckpt_path = checkpoints[0]
    print(f"loading from {ckpt_path}")
    params = MODEL_PARAMS[model_size]

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = int(math.ceil(tokenizer.n_words / vocab_align_size) * vocab_align_size)
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.forward_all = MethodType(forward_all, model)
    local_rank = int(os.environ.get("RANK", -1))
    with safe_open(ckpt_path, framework="pt") as f:
        for key, param in model.named_parameters():
            #tensor_slice = f.get_slice(key)[:]
            tensor_slice = f.get_slice(key)
            #for dim, size in enumerate(tensor_slice.size()):
            s = []
            for dim, size in enumerate(tensor_slice.get_shape()):
                if param.size(dim) != size:
                    # should chunk
                    s.append(slice(param.size(dim) * local_rank, param.size(dim) * (local_rank + 1)))
                    param.data.copy_(tensor_slice[s])
                    break
                else:
                    s.append(slice(None))
            else:
                param.data.copy_(tensor_slice[:])
    torch.set_default_tensor_type(torch.FloatTensor)

    generator = Llama(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def load_orin(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    model_size: int,
) -> Llama:
    # only used for LLaMa 65B with pth checkpoint
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = Llama(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)

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
    results = generator.generate(prompts, max_gen_len=256, temperature=temperature, top_p=top_p)

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    ckpt_dir = "/data/weights/llama/7B"
    tokenizer_path = "/data/weights/llama/tokenizer.model"
    main(ckpt_dir, tokenizer_path)
