import json
import os

from omni.eval.vqa.Accuracy_ANLS_Eval import VQAEval
from omni.eval.vqa.caption_eval import evaluate_on_coco_caption
from omni.utils.loguru import logger
from omni.utils.eval_utils import (
    clear_outputs,
    get_parser,
    merge_outputs,
    metric_mapping,
    pope_eval,
    process_mmvet,
    task_mapping,
    test_ocr,
    to_xlsx,
)

parser = get_parser()
args = parser.parse_args()


if args.evaltype == "all" or args.evaltype == "predict":
    # --------------- First step: multi-GPU evaluation --------------------
    clear_outputs(args.out_path)
    # fmt: off
    os.system(
        "python -m omni.eval.vqa.multi_hardware_eval" + " "
        + "--model_name" + " " + str(args.model_name) + " "
        + "--gtfile_path" + " " + str(args.gtfile_path) + " "
        + "--image_path" + " " + str(args.image_path) + " "
        + "--out_path" + " " + str(args.out_path) + " "
        + "--num-chunks" + " " + str(args.num_chunks) + " "
        + "--datatype" + " " + str(args.datatype) + " "
        + "--img_aug" + " " + str(args.img_aug) + " "
        + "--beamsearch" + " " + str(args.beamsearch) + " "
        + "--prompt" + " " + str(args.prompt) + " "
        + "--post_prompt" + " " + str(args.post_prompt) + " "
        + "--system_prompt" + " " + str(args.system_prompt) + " "
        + "--task_type" + " " + str(task_mapping[args.datatype]) + " "
        + "--clip" + " " + str(args.clip) + " "
    )
    # fmt: on

    # --------------- Second step: merge all results of each GPU ----------
    logger.info("Merging evaluation results of each GPU")
    merge_outputs(args.out_path)


if args.evaltype == "all" or args.evaltype == "eval":
    # --------------- Third step: evaluation metrics ----------------------
    logger.info("Start Evaluating.....")

    if task_mapping[args.datatype] == "mm":
        json_path = args.out_path + "/results_final.json"
        xlsx_path = args.out_path + "/results_final.xlsx"
        to_xlsx(json_path, xlsx_path)
    elif args.datatype == "mmvet":
        pass
    else:
        qas = json.load(open(args.out_path + "/results_final.json", encoding="utf-8"))
        gt_qa = {str(ann["question_id"]): ann["gt"] for ann in qas}
        pre_qa = {str(ann["question_id"]): ann["answer"] for ann in qas}

    if args.datatype == "mmvet":
        process_mmvet(args.gtfile_path, args.out_path)
        logger.info("MM-Vet inference is done. Please check results_final.json in the output folder.")
        logger.info("Submit to: https://huggingface.co/spaces/whyu/MM-Vet_Evaluator for evaluation.")
        acc = None

    elif task_mapping[args.datatype] == "vqa" or task_mapping[args.datatype] == "ANLS":
        # n is precision of accuracy (number of places after decimal), default is 2
        vqaEval = VQAEval(gt_qa, pre_qa, args.datatype, n=2)
        acc = vqaEval.evaluate()

    elif task_mapping[args.datatype] == "caption":
        coco_format_json = [{"image_id": qa["question_id"], "caption": qa["answer"]} for qa in qas]
        coco_format_file = args.out_path + "/results_final_coco_format.json"
        with open(coco_format_file, "w", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(coco_format_json))
        acc = evaluate_on_coco_caption(coco_format_file, f"data/eval_format_files/{args.datatype}_gt_coco_format.json")["CIDEr"] * 100

    elif task_mapping[args.datatype] == "POPE":
        acc, precision, recall, f1, yes_ratio = pope_eval(pre_qa, gt_qa)

        result_path = args.out_path + "/result_final_{}.json".format(str(args.datatype))
        with open(result_path, "w") as f:
            result = {}
            result[
                str(args.datatype)
            ] = "{} evaluation is:\n acc:{} \n precision:{} \n recall:{} \n f1:{} \n yes_ratio:{}".format(
                args.datatype, acc, precision, recall, f1, yes_ratio
            )
            json.dump(result, f, ensure_ascii=False)

    elif task_mapping[args.datatype] == "OCR":
        correct, num = test_ocr(gt_qa, pre_qa)
        acc = float(correct) / num

        logger.info("Correct: {}, Num: {}".format(correct, acc))

    if acc is not None:
        logger.info("{} evaluation {} is {}:".format(args.datatype, metric_mapping[args.datatype], acc))
