# LLaMA Evaluation

1. Install
```
export EVAL_ROOT=/path/to/dataset,database,logfiles
source environ.sh
pip install git+https://github.com/facebookresearch/llama.git
pip install -v -e .
```

2. Usage
```
python3 -m llama_evaluation.tasks.multich --tasks "ceval" --addr [ip:port] --model [MODEL_NAME] --ckpt_dir [MODEL_PATH] --sft --online --no_db
# If you want to use a MLLM, please set the MLLM's `--model` to `--ckpt_dir`
```

3. Showcasing
```
python -m streamlit run fe_display.py

```
