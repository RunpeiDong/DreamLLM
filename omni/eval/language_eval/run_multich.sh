MLP_WORKER_GPU=1
MODEL_SIZE='7b'
CKPT_DIR='path2model/llama-vicuna-7b-v1.1'
MODEL_NAME='vicuna7b'
#    --task piqa,hellaswag,winogrande,triviaqa,boolq,squad,naturalqa,race_m,race_h,quac \
# triviaqa,squad (fix)
#!/usr/bin/env bash                                                                                                                                                                                 2
torchrun --nproc_per_node $MLP_WORKER_GPU dreamllm/eval/language_eval/eval_multich.py \
    --model $MODEL_NAME \
    --model_size $MODEL_SIZE \
    --ckpt_dir $CKPT_DIR \
    --task siqa \
    --mode hf \
    --no_db