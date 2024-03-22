from omni.utils.import_utils import is_volc_mlplatform_available

DEBUG = False
# DEBUG = True
LOGDIR = "./work_dirs/"
USE_HF_LOCAL_FILES = True

HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"

# fmt: off
MODEL_ZOOS = {
    "decapoda-research/llama-7b-hf"           : "huggingface_cache/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348",
    "meta-llama/Llama-2-7b-hf"                : "huggingface_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9",
    "meta-llama/Llama-2-7b-chat-hf"           : "huggingface_cache/hub/llama2-7b-chat",
    "meta-llama/Llama-2-70b-hf"               : "huggingface_cache/hub/models--meta-llama--Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf"          : "huggingface_cache/hub/models--meta-llama--Llama-2-70b-hf",
    "lmsys/vicuna-7b-v1.1"                    : "huggingface_cache/hub/llama-vicuna-7b-v1.1",
    "lmsys/vicuna-7b-v1.3"                    : "huggingface_cache/hub/llama-vicuna-7b-v1.3",
    "lmsys/vicuna-7b-v1.5"                    : "huggingface_cache/hub/models--lmsys--vicuna-7b-v15",
    "lmsys/vicuna-13b-v1.3"                   : "huggingface_cache/hub/llama-vicuna-13b-v1.3",
    "lmsys/vicuna-33b-v1.3"                   : "huggingface_cache/hub/models--lmsys--vicuna-33b-v1.3",
    "runwayml/stable-diffusion-v1-5"          : "huggingface_cache/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9",
    "stabilityai/stable-diffusion-2-1-base"   : "huggingface_cache/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/dcd3ee64f0c1aba2eb9e0c0c16041c6cae40d780",
    "stabilityai/stable-diffusion-xl-base-0.9": "huggingface_cache/hub/models--stabilityai--stable-diffusion-xl-base-0.9/snapshots/ccb3e0a2bfc06b2c27b38c54684074972c365258",
    "stabilityai/stable-diffusion-xl-base-1.0": "huggingface_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/bf714989e22c57ddc1c453bf74dab4521acb81d8",
    "madebyollin/sdxl-vae-fp16-fix"           : "huggingface_cache/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/4df413ca49271c25289a6482ab97a433f8117d15",
    "openai/clip-vit-large-patch14"           : "huggingface_cache/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff",
    "lllyasviel/control_v11p_sd15_canny"      : "huggingface_cache/hub/models--lllyasviel--control_v11p_sd15_canny/snapshots/115a470d547982438f70198e353a921996e2e819",
    "lllyasviel/control_v11p_sd15_openpose"   : "huggingface_cache/hub/models--lllyasviel--control_v11p_sd15_openpose/snapshots/9ae9f970358db89e211b87c915f9535c6686d5ba",
    "lllyasviel/control_v11f1p_sd15_depth"    : "huggingface_cache/hub/models--lllyasviel--control_v11f1p_sd15_depth/snapshots/539f99181d33db39cf1af2e517cd8056785f0a87",
    "lllyasviel/control_v11f1e_sd15_tile"     : "huggingface_cache/hub/models--lllyasviel--control_v11f1e_sd15_tile/snapshots/3f877705c37010b7221c3d10743307d6b5b6efac",
    "openMUSE/vqgan-f16-8192-laion"           : "huggingface_cache/hub/models--openMUSE--vqgan-f16-8192-laion/snapshots/715ba7514777e2cc99d93b5ec2da8d5fd129d692",
    "openMUSE/maskgit-vqgan-imagenet-f16-256" : "huggingface_cache/hub/models--openMUSE--maskgit-vqgan-imagenet-f16-256/snapshots/9cd8b30929e5d3134e349b0ee529a43d2ae7c945",
    "openai/consistency-decoder"              : "huggingface_cache/hub/models--openai--consistency-decoder/snapshots/63b7a48896d92b6f56772f4111d0860b1bee3dd3",
}
if not is_volc_mlplatform_available():
    MODEL_ZOOS = {key: key if "/" in key else value for key, value in MODEL_ZOOS.items()}
# fmt: on

MODEL_ZOOS["fid_weights"] = (
    "huggingface_cache/pt_inception-2015-12-05-6726825d.pth"
    if is_volc_mlplatform_available()
    else "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth",
)

# model config
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION = True
IGNORE_INDEX = -100
MAX_TOKEN_LENGTH = 2048
LLM_HIDDEN_DIM = 4096
MM_HIDDEN_DIM = 1024
LDM_HIDDEN_DIM = 1024
REORDER_ATTENTION = False
VISION_HIDDEN_DIM = 256
NUM_DREAM_QUERIES = 64

# special token
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PAD_TOKEN = "[PAD]"

# additional special token
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IMAGE_START_TOKEN = "<im_start>"
DEFAULT_IMAGE_END_TOKEN = "<im_end>"

DEFAULT_DREAM_TOKEN = "<dream>"
DEFAULT_DREAM_START_TOKEN = "<dream_start>"
DEFAULT_DREAM_END_TOKEN = "<dream_end>"

# worker config
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
