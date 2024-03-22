import cv2
import numpy as np
import PIL.Image
import torch
from transformers import LlamaTokenizer

from omni.constants import MODEL_ZOOS
from omni.models.dreamllm_sdxl.configuration_dreamllm_sdxl import DreamLLMSDXLConfig as DreamLLMConfig
from omni.models.dreamllm_sdxl.modeling_dreamllm_sdxl import DreamLLMSDXLForCausalMLM as DreamLLMForCausalMLM
from omni.utils.image_utils import load_image, save_image
from omni.utils.profiler import FunctionProfiler

# download an image
image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
image = np.array(image)

# get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = PIL.Image.fromarray(image)

model_name_or_path = "path2model"

tokenizer = LlamaTokenizer.from_pretrained(
    MODEL_ZOOS["lmsys/vicuna-7b-v1.5"],
    local_files_only=True,
    padding_side="left",
)

config = DreamLLMConfig.from_pretrained(
    model_name_or_path,
    local_files_only=True,
)
config = config.reset_plugins_init_kwargs()
with FunctionProfiler("DreamLLMForCausalMLM.from_pretrained"):
    model = DreamLLMForCausalMLM.from_pretrained(
        model_name_or_path,
        tokenizer=tokenizer,
        config=config,
        local_files_only=True,
        reset_plugin_model_name_or_path=True,
    ).cuda()

# model = torch.compile(model)

prompt = [
    "a photo of an astronaut",
    "a photo of a dog",
    "a photo of a cat",
    # "a photo of an astronaut riding a horse on mars",
    "a photo of a house in beach",
    "a beach with sunset",
    "a photo of a dog is eating a burger on the moon",
]

images = model.stable_diffusion_pipeline(
    tokenizer=tokenizer,
    prompt=prompt,
    guidance_scale=0.0,
    num_inference_steps=100,
)
for i, image in enumerate(images):
    save_image(images[i], path=f"samples/{i}.jpg", force_overwrite=False)


# images = model.controlnet_pipeline(
#     tokenizer=tokenizer,
#     prompt=prompt,
#     image=canny_image,
#     guidance_scale=7.5,
#     num_inference_steps=100,
# )

# for i, image in enumerate(images):
#     save_image(images[i], path=f"{i}.jpg", force_overwrite=False)
