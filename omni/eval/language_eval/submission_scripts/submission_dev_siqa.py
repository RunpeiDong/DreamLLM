import re
import os
import time
import torch
import argparse
import jsonlines

from tqdm import tqdm
from omni.utils.loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria

from llama_evaluation_main.llama_evaluation.evaluator.evaluator_utils import (
    hf_forward,
    get_logprob_by_tokens_hf,
    exact_match,
    include_answer,
    f1_score,
    post_process,
    write_jsonl,
)

BASE_TEMPLATE = "Problem:\n{}\n\nSolution:\n"
BASE_FEWSHOT_TEMPLATE = "Problem:\n{}\n\nSolution:\n{}\n\n"
SFT_TEMPLATE = "Human: {}\n\nAssistant:"
SFT_FEWSHOT_TEMPLATE = "Human: {}\n\nAssistant: {}</t>\n\n"

SYSTEM_PROMPT = [
    "Answer each question using information in the preceding context and given options."
    'Only one of the options is correct. Only answer A, B, or C.'
]

ANSWER_PROMPT = " Human: Which option is correcnt?\n\n Answer: The correct option is {OPTION}"


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len :], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


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
    token = get_choice_tokens(tokens, tokenizer.encode(token, bos_token=False, eos_token=False))
    logits = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
    logits = torch.gather(logits[:, -token.size(1) - 1 : -1], dim=-1, index=token.unsqueeze(-1))
    return logits


def make_test_prompt(sample):
    context = "Read the context. Choose A, B or C to answer the question.\n\n"
    s = context + f'Context: {sample["context"]}\nQuestion: {sample["question"]}\nAnswer:'
    return s, [sample["answerA"], sample["answerB"], sample["answerC"]]


def make_test_prompt_sft(sample):
    s = f'Choose A, B or C to answer the question.\n\nContext: {sample["context"]}\n\nQuestion: {sample["question"]}\n\n'

    s += "Options:\n"
    s += f'A. {sample["answerA"]}\n'
    s += f'B. {sample["answerB"]}\n'
    s += f'C. {sample["answerC"]}\n'
    return s


def evaluate_siqa_generate(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, add_bos_token=False, model_max_length=4096, padding_side="right", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task)):
        os.makedirs(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task))

    with open(args.test_label_path, "r", encoding="utf-8") as f:
        labels = f.readlines()

    stop_str = "</s>"
    keywords = [stop_str]

    total, correct = 0.0, 0.0
    tic = time.time()
    logger.info("Starting evaluation SIQA dev on {} samples".format(len(labels)))
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(jsonlines.Reader(f))):
            context = make_test_prompt(item)
            context += ANSWER_PROMPT
            context = SYSTEM_PROMPT[0] + context
            # context, choices = make_test_prompt(item)
            correct_end = labels[idx]
            len_ch = 3

            question_tokens = tokenizer(context, return_tensors="pt").input_ids
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, question_tokens)
            with torch.no_grad():
                raw_results = model.generate(question_tokens, do_sample=False, temperature=0., max_new_tokens=100, stopping_criteria=[stopping_criteria])
            raw_results = tokenizer.decode(raw_results[0].tolist(), skip_special_tokens=True)
            # print(raw_results)

            answer_results = raw_results[len(context):]
            print(answer_results)

            # pred = []
            # for j in range(len_ch):
            #     ending_tokens = tokenizer(choices[j], return_tensors="pt").input_ids
            #     input_tokens = torch.cat((question_tokens, ending_tokens), dim=1).cuda()
            #     with torch.no_grad():
            #         outputs = model(input_tokens)  # Get model output
            #     logits = outputs.logits.cpu().float()
            #     logits = get_logprob_by_tokens_hf(logits, ending_tokens)
            #     pred.append(logits.sum() / len_ch)

            # predicted_end = pred.index(max(pred)) + 1  # !!! NOTE the label in siqa is 1-indexed
            # with open(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task, "results.txt"), "a") as f:
            #     f.write("{}\n".format(predicted_end))
            # Update the number of correct predictions
            # if predicted_end == int(correct_end):
            #     correct += 1
            total += 1  # Update the total number of predictions
            if total % 10 == 0:
                # Print the accuracy
                logger.info("Accuracy({}/{}): {}".format(correct, total, correct / total), end="\r")
    logger.info("Accuracy({}/{}): {}".format(correct, total, correct / total))

    toc = time.time()
    logger.info("Evaluation finished in {} seconds".format(toc - tic))
    logger.info("Done")


def evaluate_siqa(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, add_bos_token=False, model_max_length=4096, padding_side="right", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task)):
        os.makedirs(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task))

    with open(args.test_label_path, "r", encoding="utf-8") as f:
        labels = f.readlines()

    total, correct = 0.0, 0.0
    tic = time.time()
    logger.info("Starting evaluation SIQA dev on {} samples".format(len(labels)))
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(jsonlines.Reader(f))):
            context, choices = make_test_prompt(item)
            correct_end = labels[idx]
            len_ch = len(choices)
            question_tokens = tokenizer(context, return_tensors="pt").input_ids
            pred = []
            for j in range(len_ch):
                ending_tokens = tokenizer(choices[j], return_tensors="pt").input_ids
                input_tokens = torch.cat((question_tokens, ending_tokens), dim=1).cuda()
                with torch.no_grad():
                    outputs = model(input_tokens)  # Get model output
                logits = outputs.logits.cpu().float()
                logits = get_logprob_by_tokens_hf(logits, ending_tokens)
                pred.append(logits.sum() / len_ch)

            predicted_end = pred.index(max(pred)) + 1  # !!! NOTE the label in siqa is 1-indexed
            # with open(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task, "results.txt"), "a") as f:
            #     f.write("{}\n".format(predicted_end))
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
        "--test_data_path",
        type=str,
        default="./data/language_official_benchmarks/socialiqa-train-dev/dev.jsonl",
    )
    parser.add_argument(
        "--test_label_path",
        type=str,
        default="./data/language_official_benchmarks/socialiqa-train-dev/dev-labels.lst",
    )
    parser.add_argument("--task", "-t", type=str, default="siqa")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="llama-vicuna-7b")
    # parser.add_argument("--model_dir", type=str, default="path2model/llama-vicuna-7b-v1.1")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="path2model",
    )
    args = parser.parse_args()
    evaluate_siqa(args)
    # evaluate_siqa_generate(args)
