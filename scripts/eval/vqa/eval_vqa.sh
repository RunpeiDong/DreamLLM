# -------------------- Accuracy Metrics Tasks --------------------
# CoCO Caption test
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_coco_caption_test.json \
#     --image_path ./data/COCO/val2014 \
#     --out_path ./samples/eval_results/coco_cap \
#     --num-chunks 8 \
#     --datatype COCO-Captions \
#     --prompt 'Please summarize object in one sentence within 10 words.' \
#     --post_prompt "The image depicts" \
#     --evaltype "all" \
#     --img_aug none \
#     --beamsearch "True" \
#     --system_prompt "Based on the image, give the image caption briefly." \

# NoCaps
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_nocaps_caption.json \
#     --image_path ./data/nocaps/images \
#     --out_path ./samples/eval_results/nocaps \
#     --num-chunks 8 \
#     --datatype NoCaps \
#     --prompt 'Please summarize object briefly in one sentence within 5 words.' \
#     --evaltype "all" \
#     --img_aug none \
#     --beamsearch True \

# Image2Paragraph
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_image_paragraph.json \
#     --image_path ./data/paragraph-captioning/VG_100K \
#     --out_path ./samples/eval_results/image_para \
#     --num-chunks 8 \
#     --datatype Image-Paragraph \
#     --prompt 'Please describe the image in detail.' \
#     --post_prompt "The image depicts" \
#     --system_prompt "Based on the image, please describe the image in detail." \
#     --evaltype "all" \
#     --img_aug none \
#     --beamsearch True \

# VQAv2
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_VQAv2_sub_val.json \
#     --image_path ./data/COCO/val2014 \
#     --out_path ./samples/eval_results/vqav2 \
#     --num-chunks 4 \
#     --datatype VQAv2 \
#     --img_aug none \
#     --beamsearch True \
#     --evaltype "all" \
#     # --system_prompt "Based on the image, please answer the question." \
#     # --clip True \
#     # --prompt "Please provide an accurate answer within one word or phrase." \
#     # --post_prompt "The answer is:" \

# VQAv2-test-dev
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_VQAv2_test_dev.json \
#     --image_path ./data/VQAv2/test2015 \
#     --out_path ./samples/eval_results/vqav2 \
#     --num-chunks 8 \
#     --datatype VQAv2 \
#     --img_aug none \
#     --beamsearch True \
#     --system_prompt "Based on the image, please answer the question." \
#     --prompt "Please provide an accurate answer within one word." \
#     --post_prompt "The short answer '\(one word\)' is:" \
#     --clip True
    # --post_prompt "The answer is:" \

# TextVQA
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model  \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_TextVQA_val.json \
#     --image_path ./data/TextVQA/train_images \
#     --out_path ./samples/eval_results/textvqa \
#     --num-chunks 8 \
#     --datatype TextVQA \
#     --img_aug none \
#     --beamsearch True \
#     --evaltype "eval" \
#     --system_prompt "Based on the image, please answer the question." \
#     --prompt "Please provide an accurate answer within one word." \
#     --post_prompt "The answer is:" \
#     --clip True

# MMBench
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/mmbench_dev_20230712.tsv \
#     --image_path none \
#     --out_path ./samples/eval_results/mmbench \
#     --num-chunks 2 \
#     --datatype mmbench \
#     --img_aug none \
#     --beamsearch True \
#     --evaltype "all" \
#     --prompt "Please provide an accurate and detailed answer." \
#     --system_prompt "This is an exam, please answer according to the image, hint and question."

# MM-Vet (Submit to https://huggingface.co/spaces/whyu/MM-Vet_Evaluator)
# python -m omni.eval.vqa.eval_dreamllm \
#     --gtfile_path ./data/MM-VET/MMGPT_mm-vet.json \
#     --model_name path2model \
#     --image_path ./data/MM-VET/images \
#     --out_path ./samples/mm-vet \
#     --num-chunks 8 \
#     --datatype mmvet \
#     --img_aug none \
#     --beamsearch True \
#     --evaltype "all" \
#     --prompt "Please provide an accurate, detailed and comprehensive answer." \
#     --system_prompt "This is an exam, please answer according to the image and question."

# VizWizVQA
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_VizWizVQA_val.json \
#     --image_path ./data/VizWiz-VQA/val \
#     --out_path ./samples/eval_results/VizWizVQA \
#     --num-chunks 8 \
#     --datatype VizWizVQA \
#     --img_aug none \
#     --beamsearch True \

# VizWizVQA-test
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_VizWizVQA_test.json \
#     --image_path ./data/VizWiz-VQA/test \
#     --out_path ./samples/eval_results/VizWizVQA \
#     --num-chunks 8 \
#     --datatype VizWizVQA \
#     --img_aug none \
#     --beamsearch True \
#     --prompt "Please provide an accurate answer within one word." \
#     --post_prompt "The answer is:" \
#     --system_prompt "Based on the image, please answer the question." \
#     --clip True

# OKVQA
python -m omni.eval.vqa.eval_dreamllm \
    --model_name path2model \
    --gtfile_path ./data/eval_format_files/OMNI_format_OKVQA_val.json \
    --image_path ./data/COCO/val2014 \
    --out_path ./samples/OKVQA/ \
    --num-chunks 8 \
    --datatype OKVQA \
    --img_aug none \
    --beamsearch True \
    --prompt "Please provide an accurate and brief answer within one word." \
    --post_prompt "The short answer '\(one word\)' is:" \
    --evaltype "all" \
    --clip True


# -------------------- ANLS Metrics Tasks --------------------

# DocVQA
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_DocVQA_val.json \
#     --image_path ./data/DocVQA/val/documents \
#     --out_path ./samples/eval_results/DocVQA \
#     --num-chunks 8 \
#     --datatype DocVQA \
#     --img_aug padding_square_resize \
#     --beamsearch True \

# InfographicVQA
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_InfographicsVQA_val.json \
#     --image_path ./data/InfographicsVQA/infographicVQA_val_v1.0_images \
#     --out_path ./samples/eval_results/InfographicVQA \
#     --num-chunks 8 \
#     --datatype InfographicVQA \
#     --img_aug padding_square_resize \
#     --beamsearch True \


# -------------------- OCR Tasks ---------------------
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_OCR_val.json \
#     --image_path ./data/OCR/IC13_857 \
#     --out_path ./samples/eval_results/OCR \
#     --num-chunks 8 \
#     --datatype OCR \
#     --img_aug padding_square_resize \
#     --beamsearch True \


# -------------------- POPE Tasks --------------------
# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_coco_pope_random.json \
#     --image_path ./data/COCO/val2014 \
#     --out_path ./samples/eval_results/POPE_random \
#     --num-chunks 8 \
#     --img_aug none \
#     --beamsearch False \
#     --datatype POPE_random \
#     --evaltype all \
#     --system_prompt "Based on the image, please objectively and accurately indicate whether the object exists." \
#     --post_prompt "The answer is:" \

# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_coco_pope_popular.json \
#     --image_path ./data/COCO/val2014 \
#     --out_path ./samples/eval_results/POPE_popular \
#     --num-chunks 8 \
#     --img_aug none \
#     --beamsearch False \
#     --datatype POPE_popular \
#     --system_prompt "Based on the image, please objectively and accurately indicate whether the object exists." \
#     --post_prompt "The answer is:" \

# python -m omni.eval.vqa.eval_dreamllm \
#     --model_name path2model \
#     --gtfile_path ./data/omni_comprehension_eval_format_files/OMNI_format_coco_pope_adversarial.json \
#     --image_path ./data/COCO/val2014 \
#     --out_path ./samples/eval_results/POPE_adversarial \
#     --num-chunks 8 \
#     --img_aug none \
#     --beamsearch False \
#     --datatype POPE_adversarial \
#     --system_prompt "Based on the image, please objectively and accurately indicate whether the object exists." \
#     --post_prompt "The answer is:" \
