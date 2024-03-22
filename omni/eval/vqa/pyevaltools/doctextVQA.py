import argparse
import json

from omni.eval.vqa.pyevaltools.doctextVQAeval import VQAEval

parser = argparse.ArgumentParser()
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--gt_path", type=str, required=True)
parser.add_argument("--datatype", type=str, required=True)
args = parser.parse_args()


def doc_text_eval(gt_root_, predict_root_, datatype):
    gts = json.load(open(gt_root_, encoding="utf-8"))

    predicts = json.load(open(predict_root_, encoding="utf-8"))

    try:
        gt_qa = {ann["question_id"]: [] for ann in gts["data"]}
        flag = 0
    except:
        gt_qa = {ann["questionId"]: [] for ann in gts["data"]}
        flag = 1

    for ann in gts["data"]:
        if flag == 1:
            gt_qa[ann["questionId"]] = ann
        else:
            gt_qa[ann["question_id"]] = ann

    pre_qa = {ann["question_id"]: [] for ann in predicts}

    for ann in predicts:
        pre_qa[ann["question_id"]] = ann

    vqaEval = VQAEval(gt_qa, pre_qa, datatype, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    # vqaEval = VQAEval(gts, predicts, datatype, n=2)
    if datatype == "Doc":
        annls = vqaEval.evaluate()
        print("DocVQA evaluation ANLS is:", annls)
    else:
        acc = vqaEval.evaluate()
        print("Overall Accuracy is:", acc)


# doc_text_eval("/data/data/DocVQA/val/val_v1.0.json", "/data/codes/llava-main/results_cc595k-freeze-docvqa-unfreeze-224/results_final.json", "Doc")


doc_text_eval(args.gt_path, args.out_path + "/results_final.json", args.datatype)
