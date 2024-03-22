import re
import os
import time
import torch
import argparse
import jsonlines

from tqdm import tqdm
from omni.utils.loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_TEMPLATE = "Problem:\n{}\n\nSolution:\n"
BASE_FEWSHOT_TEMPLATE = "Problem:\n{}\n\nSolution:\n{}\n\n"
SFT_TEMPLATE = "Human: {}\n\nAssistant:"
SFT_FEWSHOT_TEMPLATE = "Human: {}\n\nAssistant: {}</t>\n\n"


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    text = text.replace("..", ".")
    return text.strip()


def get_logprob_by_tokens_hf(logits, token):
    logits = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
    logits = torch.gather(logits[:, -token.size(1) - 1 : -1], dim=-1, index=token.unsqueeze(-1))
    return logits


def evaluate_hellaswag(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, add_bos_token=False, model_max_length=4096, padding_side="right", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task)):
        os.makedirs(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task))

    tic = time.time()
    logger.info("Starting evaluation")
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for item in tqdm(jsonlines.Reader(f)):
            # prompt_render = get_train_prompt(args.train_data_dir, args.ntrain)
            context = item["activity_label"] + ": " + item["ctx_a"] + " " + item["ctx_b"].capitalize()
            context = preprocess(context)
            endings = item["ending_options"]  # Possible endings
            question_tokens = tokenizer(context, return_tensors="pt").input_ids
            # Generate a score for each possible ending
            losses = []
            pred = []
            for ending in endings:
                ending = preprocess(ending)
                ending_tokens = tokenizer(ending, return_tensors="pt").input_ids
                input_tokens = torch.cat((question_tokens, ending_tokens), dim=1).cuda()
                with torch.no_grad():
                    outputs = model(input_tokens)  # Get model output

                logits = outputs.logits.cpu().float()
                logits = get_logprob_by_tokens_hf(logits, ending_tokens)
                pred.append(logits.sum() / len(ending))

            predicted_end = pred.index(max(pred))
            with open(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task, "results.txt"), "a") as f:
                f.write(
                    "{}\n".format(predicted_end)
                )

    toc = time.time()
    logger.info("Evaluation finished in {} seconds".format(toc - tic))
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--ngpu", "-g", type=int, default=8)
    parser.add_argument(
        "--test_data_path",
        "-d",
        type=str,
        default="./data/language_official_benchmarks/hellaswag-test/hellaswag.jsonl"
    )
    parser.add_argument("--task", "-t", type=str, default="hellaswag")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="llama-vicuna-7b")
    # parser.add_argument("--model_dir", type=str, default="path2model/llama-vicuna-7b-v1.1")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="path2model",
    )
    args = parser.parse_args()
    evaluate_hellaswag(args)
