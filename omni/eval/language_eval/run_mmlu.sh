MLP_WORKER_GPU=1
MODEL_SIZE='7b'
# CKPT_DIR='path2model/llama-vicuna-7b-v1.1'
CKPT_DIR="path2model"
DATA_DIR='./data/mmlu/'

torchrun --nproc_per_node $MLP_WORKER_GPU dreamllm/eval/language_eval/evaluate_mmlu.py \
	--ntrain 5 \
	--data_dir $DATA_DIR \
	--model $CKPT_DIR


#--ckpt_dir /data/code/ChatDreamer-Private/work_dirs_interleave_text_image/checkpoint-1000 \
# torchrun --nproc_per_node $MLP_WORKER_GPU dreamllm/eval/language_eval/eval_mmlu.py \
# 	--model_size $MODEL_SIZE \
# 	--data_dir $DATA_DIR \
# 	--model vicuna_7b \
# 	--ckpt_dir $CKPT_DIR \
# 	--language en \
# 	--max_seq_len 1024
