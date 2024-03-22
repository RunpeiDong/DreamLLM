import re
import readline  # Importing this package will make deletion normal

import torch
from transformers import LlamaTokenizer

from omni.conversation.multimodal import ModalType, MultimodalContent, Unimodal
from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM
from omni.utils.image_utils import load_image, save_image
from omni.utils.loguru import logger
from omni.utils.profiler import FunctionProfiler


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
        with FunctionProfiler("DreamLLMForCausalMLM.from_pretrained"):
            model = DreamLLMForCausalMLM.from_pretrained(
                model_name_or_path,
                tokenizer=tokenizer,
                config=config,
                local_files_only=True,
            )
    return tokenizer, model


def parse_prompt(prompt):
    params_pattern = r"\{([^}]+)\}"

    default_params = {"cfg": 7.5, "steps": 100}
    params = {}

    params_match = re.search(params_pattern, prompt)

    if params_match:
        params_string = params_match.group(1)
        for param in params_string.split(","):
            key, value = param.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "image":
                if "<image>" not in prompt:
                    raise ValueError("The prompt must contain an <image> tag.")
                params[key] = value
            elif key == "cfg":
                params[key] = float(value)
            elif key == "steps":
                params[key] = int(value)

    for key, value in default_params.items():
        params.setdefault(key, value)

    prompt_cleaned = re.sub(params_pattern, "", prompt).strip()

    return prompt_cleaned, params


if __name__ == "__main__":
    """
    params format: {image: path/to/image, cfg: 7.5, steps: 100}
    concat with a normal prompt (e.g. 'a photo of dog', 'a photo of <image> and a dog')
    """
    model_name_or_path = input(
        "Enter the first model you want to be tested, and then you can modify the model by entering 'model_name_or_path=xxx':\n"
    )
    tokenizer, model = load_model(model_name_or_path)
    while True:
        try:
            user_input = input("Enter a prompt or a new model path: ")
            if user_input.startswith("model_name_or_path="):
                model_name_or_path = user_input.split("=")[1].strip()
                tokenizer, model = load_model(model_name_or_path)
                logger.info(f"Model updated to: {model_name_or_path}")
            else:
                prompt, params = parse_prompt(user_input)
                if "image" in params:
                    path = params["image"]
                    prompt = MultimodalContent(
                        text=prompt,
                        unimodal_list=[Unimodal(modal_type=ModalType.IMAGE, placeholder="<image>", content=[load_image(path)])],
                    )
                image = model.stable_diffusion_pipeline(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    guidance_scale=params["cfg"],
                    num_inference_steps=params["steps"],
                )[0]
                save_name = (
                    prompt if isinstance(prompt, str) else f"{prompt.text}_{params['image'].split('.')[0].replace('/', '_')}"
                )
                save_image(image, path=f"./samples/{model_name_or_path}/{save_name}.jpg", force_overwrite=False)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(e)
            continue
