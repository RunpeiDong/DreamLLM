<div align="center">
<img src="../../assets/images/dreamllm_text.svg" style="width: 30%" alt="DreamLLM Logo"/>
</div>

<div align="center">

<!-- # Dromedary -->

### ICLR 2024 (Spotlight)

## <a href="https://openreview.net/forum?id=y01KGvd9Bw">DreamLLM: Synergistic Multimodal Comprehension and Creation</a>
</div>

# DreamLLM based on SDv2.1

## Training Notes
### 1. Stage-I Alignment Training
In this stage, we align the multimodal inputs and outputs with LLMs. Only the projectors and dream embeddings will be trained. The LLMs are frozen. To improve the training efficiency and balance the bottleneck between multimodal comprehension and creation, we split the training into `comprehension_only` and `cration_only` modes and train these two modes separately. This is because the training of these two tasks won't influence each other (no gradient is interleaved). Just set these two arguments to `True` or `False` to change the setting.
- `creation_only`: Only the dream queries and the projector that projects the LLM-encoded dream embeddings into the SD decoder will be trained. The dataset is typically an image-text pair dataset.
- `comprehension_only`: Only the projector that projects CLIP vision embeddings will be trained. The dataset is typically image-text pair datasets and the pretraining multimodal conversation datasets such as LLaVAPretrain.

**Notes:** There will be three parts of the models' weights saved. The `clip_vision_embedding.bin` file saves both the CLIP weights and the projector weights. The `dream_embedding.bin` file saves the dream query weights. The `stable_diffusion_head.bin` file saves both the SD model and the projector. The `.safetensors` save the main model weights. Therefore, we just need the `clip_vision_embedding.bin` from the `comprehension_only` model and the others from the `creation_only` model.

### 2. Stage-II $\mathcal{I}$-GPT Training
In this stage, we train DreamLLM to perform generative modeling on interleaved datasets like MMC4. All the models will be trained except for the SD decoder and CLIP encoder. We also use some image-text pair datasets to keep the generated image quality and the single-caption alignment capability.

**Notes:** Before training, you have to modify the `stage2` config file to change the `clip_vision_embedding_config_init_kwargs.pretrained_model_name_or_path`, `stable_diffusion_head_config_init_kwargs.pretrained_model_name_or_path`, and `dream_embedding_config_init_kwargs.pretrained_model_name_or_path` to the corresponding folder that saves `clip_vision_embedding.bin`, `stable_diffusion_head.bin`, and `dream_embedding.bin`, respectively.
### 3. Stage-III SFT Instruction-Following Training
In this stage, we train DreamLLM to perform SFT on instruction-following data. There are mainly three kinds of data, including image-text pairs, instruction-following interleaved generation data, and instruction-following conversation datasets.

**Notes:** Before training, you have to modify the `sft` config file to change the `model_name_or_path` to the stage-II pretrained model path. To improve the image aesthetic quality, you can try to add some images from datasets such as JourneyDB, which would significantly improve the output image style/aesthetic quality. However, better aesthetic quality typically leads to worse COCO FID.

## Inference
During inference, you can load the model by:
```python
from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM
from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from omni.utils.profiler import FunctionProfiler

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
        reset_plugin_model_name_or_path=True, # NOTE: Don't forget to reset.
    ).cuda()
```
