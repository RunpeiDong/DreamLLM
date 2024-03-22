import os

import cv2
import numpy as np
import PIL.Image
import torch
from accelerate.utils import set_seed
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizer

from omni.constants import MODEL_ZOOS
from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM as AutoModelForCausalLM
from omni.utils.image_utils import load_image, save_image
from omni.utils.profiler import FunctionProfiler

set_seed(42, device_specific=False)

if not os.path.isdir("samples/images/"):
    os.makedirs("samples/images")

# This is for DreamLLM-Controlnet, which is not fully supported at the present
# download an image
# image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
# image = np.array(image)

# # get canny image
# image = cv2.Canny(image, 100, 200)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = PIL.Image.fromarray(image)

# NOTE: download the model to the specified local directory
# FIXME: directly download to the huggingface cache would possibly lead to wrong model loading (bug)
model_name_or_path = "huggingface_cache/dreamllm-7b-chat-aesthetic-v1.0"
snapshot_download("RunpeiDong/dreamllm-7b-chat-aesthetic-v1.0", local_dir=model_name_or_path)

local_files_only = False

tokenizer = LlamaTokenizer.from_pretrained(
    model_name_or_path,
    local_files_only=local_files_only,
    padding_side="right",
)

config = DreamLLMConfig.from_pretrained(
    model_name_or_path,
    local_files_only=local_files_only,
)
with FunctionProfiler("AutoModelForCausalLM.from_pretrained"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        tokenizer=tokenizer,
        config=config,
        local_files_only=local_files_only,
        cache_dir=None,
        reset_plugin_model_name_or_path=True,
    ).to(dtype=torch.float32).cuda()
# config.update_plugins(
#     dict(
#         _class_=ControlNetHead,
#         _name_="controlnet_head",
#         _plugin_type_="head",
#         controlnet_model_name_or_path=MODEL_ZOOS["lllyasviel/control_v11p_sd15_canny"],
#         diffusion_name_or_path=MODEL_ZOOS["runwayml/stable-diffusion-v1-5"],
#         pretrained_model_name_or_path=model_name_or_path,
#         local_files_only=True,
#     )
# )
model = torch.compile(model)

prompt = [
    "A photo of a dog.",
    "A photo of a black dog.",
    "A polar bear in the forest.",
    "A cute cat.",
    "An astronaut riding a horse.",
    "An espresso machine that makes coffee, art station.",
    "A couple of wine glasses on a dining table. DSLR photograh.",
    "Downtown Manhattan at sunrise. Detailed ink wash.",
    "Downtown Beijing at sunrise. Detailed ink wash.",
    "A cat and a glass of whisky. DSLR photograh. Close shot.",
    "an armchair in the shape of an avocado",
    "orange fruit",
    "(painting,fantasy) A tiger with a suit, fantasy painting.",
    "A polar bear by the river.",
    "A photo of a frog reading the newspaper named “Toaday” written on it.",
    "A panda is reading the newspaper."
]

positive_prompt = "(best quality,extremely detailed)"
prompt = [f"{positive_prompt} {p}" for p in prompt]
negative_prompt = "ugly,duplicate,morbid,mutilated,tranny,mutated hands,poorly drawn hands,blurry,bad anatomy,bad proportions,extra limbs,cloned face,disfigured,missing arms,extra legs,mutated hands,fused fingers,too many fingers,unclear eyes,lowers,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,bad feet,gettyimages"
negative_prompt = [negative_prompt] * len(prompt)
# negative_prompt = None

images = model.stable_diffusion_pipeline(
    tokenizer=tokenizer,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=3.5,
    num_inference_steps=150,
)
for i, image in enumerate(images):
    save_image(images[i], path=f"samples/images/{i}.jpg", force_overwrite=False)

# images = model.controlnet_pipeline(
#     tokenizer=tokenizer,
#     prompt=prompt,
#     image=canny_image,
#     guidance_scale=7.5,
#     num_inference_steps=100,
# )

# for i, image in enumerate(images):
#     save_image(images[i], path=f"{i}.jpg", force_overwrite=False)
