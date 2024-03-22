#!/bin/bash
out_dir=$1
model_name_or_path=$2

seed=(42 43 44 45 46 47 48 49)

PYTHONPATH=path2dreamllm/ \
accelerate launch --num_processes 8 omni/eval/text2img/ddp_sample_coco.py \
--type caption \
--coco_root data/coco_fid_files \
--ann_file captions_val2014.json \
--n_samples 30000 \
--batch_size_per_device 5 \
--out_data_info_path "samples/${out_dir}/data_info.json"

for ((i=0; i<=7; i++))
do
    PYTHONPATH=path2dreamllm/ \
    accelerate launch --num_processes 8 omni/eval/text2img/ddp_sample_coco.py \
    --type dreamllm \
    --model_name_or_path $model_name_or_path \
    --diffusion_model_name_or_path models--stabilityai--stable-diffusion-2-1-base/snapshots/dcd3ee64f0c1aba2eb9e0c0c16041c6cae40d780 \
    --clip_model_name_or_path models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff \
    --coco_root data/coco_fid_files \
    --ann_file captions_val2014.json \
    --n_samples 30000 \
    --batch_size_per_device 24 \
    --out_dir "samples/${out_dir}/seed${seed[i]}" \
    --num_inference_steps 150 \
    --guidance_scale 2.0 \
    --local_files_only \
    --seed ${seed[$i]}
done

PYTHONPATH=path2dreamllm/ \
accelerate launch --num_processes 8 omni/eval/text2img/ddp_sample_coco.py \
    --type select \
    --data_info_path "samples/${out_dir}/data_info.json" \
    --clip_for_similarity_model_name_or_path models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff \
    --base_dir samples/${out_dir} \
    --dirs_name seed \
    --out_dir final_res

for ((i=0; i<=7; i++))
do
    python -m pytorch_fid data/coco_fid_files/fid_stats_mscoco256_val.npz samples/${out_dir}/seed${seed[i]}
done
python -m pytorch_fid data/coco_fid_files/fid_stats_mscoco256_val.npz samples/${out_dir}/final_res
