# Evaluations

We currently support evaluation for three main kinds of tasks:
- **VQA (Comprehension)**
- **NLP (Comprehension)**
- **Text2Image (Creation)**

## VQA Comprehension
We have supported the evaluation on several datasets at `omni/eval/vqa`.
### Data Preparation
All formatted annotation files `omni_comprehension_eval_format_files` can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1olTmjeOmFTtD9Z5GXKSDOsh7eWElCtbr?usp=sharing), the formatted MM-Vet dataset `MM-VET` can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1sFUJV4NCbzZzsw4szrAN1FvoOrzpIwIB?usp=sharing). Please download them and put them under `data`.

### Supported Datasets
|     Dataset     |     Task      | Metrics  |                      Evaluation Server                       |
| :-------------: | :-----------: | :------: | :----------------------------------------------------------: |
|      VQAv2      |      VQA      | Accuarcy | [Server](https://eval.ai/web/challenges/challenge-page/830/evaluation) |
|      OKVQA      |      VQA      | Accuarcy |                            Local                             |
|     VizWiz      |      VQA      | Accuarcy | [Server](https://eval.ai/web/challenges/challenge-page/1560/evaluation) |
|     TextVQA     |      VQA      | Accuarcy |                            Local                             |
|     MM-Vet      |      VQA      | Accuarcy | [Server](*https://huggingface.co/spaces/whyu/MM-Vet_Evaluator*) |
|     MMBench     |      VQA      | Accuarcy | [Server](https://mmbench.opencompass.org.cn/mmbench-submission) |
|  COCO Caption   |  Captioning   |  CIDEr   |                            Local                             |
| Image2Paragraph |  Captioning   |  CIDEr   |                            Local                             |
|     NoCaps      |  Captioning   |  CIDEr   |                            Local                             |
|     DocVQA      |      VQA      |   ANLS   |      [Server](https://rrc.cvc.uab.es/?ch=11&com=tasks)       |
| InfographicVQA  |      VQA      |   ANLS   |      [Server](https://rrc.cvc.uab.es/?ch=11&com=tasks)       |
|      POPE       | Hallucination | Accuracy |                            Local                             |

**Notes:** DocVQA and InfographicVQA require high resolution to get a reasonable result, so a model that is trained on low-resolution images (e.g., 224x224) and uses CLIP as the vision encoder will get a very low performance. Models like [Vary](https://varytoy.github.io/) that use high-resolution images and hybrid image representations will be better at this task.

### Run
To evaluate VQA tasks such as MM-Vet, please run the following:
```shell
# MM-Vet (Submit to https://huggingface.co/spaces/whyu/MM-Vet_Evaluator)
python -m omni.eval.vqa.eval_dreamllm \
    --model_name path2model \
    --gtfile_path ./data/MM-VET/MMGPT_mm-vet.json \
    --image_path ./data/MM-VET/images \
    --out_path ./samples/mm-vet \
    --num-chunks 1 \
    --datatype mmvet \
    --img_aug none \
    --beamsearch True \
    --evaltype "all" \
    --prompt "Please provide an accurate and detailed answer." \
    --system_prompt "This is an exam, please answer according to the image and question."
```
Then, submit the file `results_final.json` to [server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) for the results.

We have provided a script, `scripts/eval/vqa/eval_vqa.sh` for testing different benchmarks. 

## NLP Comprehension
We have supported NLP evaluation on multi-task language processing and other QA datasets at `omni/eval/language_eval`.

### Data Preparation
All formatted annotation files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/18sHrGCJga64yb2iBbNfKKmn8njsF9cuW?usp=drive_link). The MMLU dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Ima5i3aV-CCpg9TayW-ylHUkaMFGXfkm?usp=drive_link).

### Supported Datasets
|  Dataset   |    Task    |              Evaluation Server               |
| :--------: | :--------: | :------------------------------------------: |
|   BoolQ    |     QA     |                    Local                     |
|    PIQA    |     QA     |                    Local                     |
|    SIQA    |     QA     |                    Local                     |
| HellaSwag  |     QA     | [Server](https://allenai.org/data/hellaswag) |
| WinoGrande |     QA     |                    Local                     |
|    MMLU    | Multi-task |                    Local                     |

### Run
We have integrated a comprehensive evaluation toolkit called `llama_evlaution_main`. This toolkit supports various dataset evaluations with huggingface, but the dataset split may be different from the official ones that are typically used in papers. For official comparison, you can run the evaluation scripts at `omni/eval/language_eval/submission_scripts`. For example, if you want to evlaute BoolQ accuracy, please run:
```shell
python omni/eval/language_eval/submission_scripts/submission_dev_boolq.py \
    --model_dir path2model
```

## Text2Image
We have supported text-to-image evaluation on COCO and LN-COCO at `omni/eval/text2img`.
### Data Preparation
- You have to first prepare the MS COCO images or the FID statistics files. The caption annotation files include `captions_train2014.json` and `captions_val2014.json` for MS COCO and `lncoco_captions_val2017.jsonl` for LN COCO. To calculate FID, you have to prepare the `fid_stats.npz` file, which is `fid_stats_mscoco256_val.npz` for MS COCO and `fid_stats_lncoco256_val5k.npz` for LN COCO. 
We have uploaded all these files in `coco_fid_files` on [Goolge Drive](https://drive.google.com/drive/folders/1wopXzbuu70hDSEgG7dzzUekurdxW-zFU?usp=sharing).
- If you have your own dataset, you can make the fid_stats file by running
```shell
python ./third_party/pytorch-fid/src/pytorch_fid/fid_score.py \
    --path "path2images" "path2fid_stats.npz" \
    --resolution 256 \
    --batch-size 50 \
    --save-stats
```
### Supported Datasets
We currently support MS COCO and LN COCO datasets.
| Dataset |    Task    | Metrics | Evaluation Server |
| :-----: | :--------: | :-----: | :---------------: |
| MS COCO | Text2Image |   FID   |       Local       |
| LN COCO | Text2Image |   FID   |       Local       |

### Run
We have provided the scripts to run COCO FID evaluation based on 8-times selection with CLIP. Just run:
```shell
OUT_DIR="OUTPUT_DIR"
MODEL_NAME_OR_PATH="YOUR_MODEL_PATH"
sh scripts/eval/text2img/eval_coco_zero_shot_clip8_select.sh $OUTPUT_DIR $MODEL_NAME_OR_PATH
```