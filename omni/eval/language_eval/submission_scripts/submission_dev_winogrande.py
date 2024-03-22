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


def get_choice_tokens(token_a, token_b):
    cur = len(token_a) - len(token_b)
    for i in range(cur, len(token_a)):
        if token_a[i] == token_b[i - cur]:
            return torch.tensor(token_b[i - cur :]).unsqueeze(0)
    return ""


def get_logprob_by_tokens(tokens, logits, tokenizer, token):
    # tokens, logits = local_forward(sample, generator)
    # import ipdb; ipdb.set_trace()
    token = get_choice_tokens(tokens, tokenizer.encode(token, bos_token=False, eos_token=False))
    logits = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
    logits = torch.gather(logits[:, -token.size(1) - 1 : -1], dim=-1, index=token.unsqueeze(-1))
    return logits


def make_test_prompt(sample):
    pronoun_loc = sample["sentence"].index("_")
    pre_ctx = sample["sentence"][:pronoun_loc].strip()
    post_ctx = " " + sample["sentence"][pronoun_loc + 1 :].strip()
    return pre_ctx, [sample["option1"], sample["option2"], post_ctx]


def evaluate_winogrande(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, add_bos_token=False, model_max_length=4096, padding_side="right", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task)):
        os.makedirs(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task))

    total, correct = 0.0, 0.0
    tic = time.time()
    logger.info("Starting evaluation")
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for item in tqdm(jsonlines.Reader(f)):
            context, choices = make_test_prompt(item)
            correct_end = item["answer"]  # Correct answer index is in 'answer' field
            post_ctx = choices.pop()
            len_ch = len(choices)

            pred = []
            for j in range(len_ch):
                sample = " ".join([context, choices[j], post_ctx])
                input_tokens = tokenizer(sample, return_tensors="pt").input_ids.cuda()
                with torch.no_grad():
                    outputs = model(input_tokens)  # Get model output
                logits = get_logprob_by_tokens(input_tokens[0], outputs.logits, tokenizer, post_ctx)
                pred.append(logits.sum())
                # pred.append(logits.sum() / len_ch)

            predicted_end = pred.index(max(pred)) + 1 # !!! NOTE the label in winogrande is 1-indexed
            with open(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task, "results.txt"), "a") as f:
                f.write("{}\n".format(predicted_end))
            # Update the number of correct predictions
            if predicted_end == int(correct_end):
                correct += 1
            total += 1  # Update the total number of predictions
            if total % 10 == 0:
                # Print the accuracy
                logger.info("Accuracy({}/{}): {}".format(correct, total, correct / total), end="\r")
    logger.info("Accuracy({}/{}): {}".format(correct, total, correct / total))

    toc = time.time()
    logger.info("Evaluation finished in {} seconds".format(toc - tic))
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--ngpu", "-g", type=int, default=8)
    parser.add_argument(
        "--test_data_path", "-d", type=str, default="./data/language_official_benchmarks/winogrande_1.1/dev.jsonl"
    )
    parser.add_argument("--task", "-t", type=str, default="winogrande")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="llama-vicuna-7b")
    # parser.add_argument("--model_dir", type=str, default="path2model/llama-vicuna-7b-v1.1")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="path2model",
    )
    args = parser.parse_args()
    evaluate_winogrande(args)
