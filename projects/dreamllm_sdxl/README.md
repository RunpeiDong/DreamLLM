<div align="center">
<img src="../../assets/images/dreamllm_text.svg" style="width: 30%" alt="DreamLLM Logo"/>
</div>

<div align="center">

<!-- # Dromedary -->

### ICLR 2024 (Spotlight)

## <a href="https://openreview.net/forum?id=y01KGvd9Bw">DreamLLM: Synergistic Multimodal Comprehension and Creation</a>
</div>

# DreamLLM based on SDXL
Please see [here](../dreamllm/README.md) to get the basic notes of DreamLLM based on SDv2.1.

## Training Notes
For SDXL, there are some details that should be noted for a more stable training.
- The VAE model in SDXL must be in FP32, otherwise the model may not converge.
- A good practice is to first train DreamLLM-SDXL on images of lower original resolution (e.g., 64-256, which will all be resized to 1024 then) or quality to quickly warm up. Then you can try images with larger resolutions and higher quality such as aesthetic standards, DreamLLM-SDXL will then quickly learn the SDXL style image generation by following instructions.