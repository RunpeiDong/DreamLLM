torchrun --master_addr $MLP_WORKER_0_HOST --master_port $MLP_WORKER_0_PORT --node_rank $MLP_ROLE_INDEX --nnodes $MLP_WORKER_NUM --nproc_per_node 8 \
-m projects.dreamllm.train \
--config_file projects/dreamllm/configs/stage2/base.py \
"training.save_steps=2000" \
"training.vit_llrd=False" \
"training.llm_llrd=False" \
"training.unfreeze_vit=False" \
"data.comprehension_only=False" \
"data.creation_only=False" \
"training.per_device_train_batch_size=16" \
"training.num_train_epochs=1" \
"training.output_dir='./work_dirs/dreamllm_vicuna11sd21__stage2/'" \
"training.learning_rate=2e-5"