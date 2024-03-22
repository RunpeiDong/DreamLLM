import re
import os
import time
import json
import torch
import argparse
import jsonlines
import numpy as np

from tqdm import tqdm
from omni.utils.loguru import logger
from conversation import default_conversation
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

SEP = "</s>"
BASE_TEMPLATE = "Problem:\n{}\n\nSolution:\n"
BASE_FEWSHOT_TEMPLATE = "Problem:\n{}\n\nSolution:\n{}\n\n"
SFT_TEMPLATE = "Human: {}\n\nAssistant:"
SFT_FEWSHOT_TEMPLATE = "Human: {}\n\nAssistant: {}</t>\n\n"

# CHECK_QUESTION = " Is the answer can be known? "
# CHECK_QUESTION = "Is the answer known?"
CHECK_QUESTION = 'Is there enough information in the background provided to answer the question? If there is not enough information provided, answer with "Not in background."\n'

SYSTEM_PROMPT = [
    "Answer each question using information in the preceding background paragraph."
    'If there is not enough information provided, answer with "Not in background."'
]

ANSWER_DEMAND = "USER: Please provide an accurate, short answer. Do not ask any more questions.\nASSISTANT: The short answer is:"

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


def short_generation_evaluator(dataset, generator, batch_size, fs):
    cors, incs, f1s = [], [], []
    questions, labels = [], []
    count = 1

    for i, (question, label) in tqdm(enumerate(dataset)):
        questions.append(question)
        labels.append(label)
        if i < (len(dataset) - 1) and (i + 1) % batch_size != 0:
            continue

        raw_results = generator.generate(questions, 100, temperature=0)
        if "\n\nAssistant:" in questions[0]:
            results = [i.split("Assistant:")[0].strip() for i in raw_results]
        else:
            results = [i.split("Answer:")[1].split("Question:")[0].strip() for i in raw_results]
        cor = [exact_match(result, label) for result, label in zip(results, labels)]
        inc = [include_answer(result, label) for result, label in zip(results, labels)]
        f1s.extend([f1_score(result, label) for result, label in zip(results, labels)])
        cors.extend(cor)
        incs.extend(inc)
        questions.clear()
        labels.clear()

    return {"include": np.mean(incs), "accuracy": np.mean(cors), "F1 score": np.mean(f1s)}


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


def make_test_prompt(sample, context, title=None):
    question_a  = "Read the passage and judge if the asked question can be answered. Only answer yes or no.\n"
    question = "Read the passage and give me a brief answer. \n"
    if title is not None:
        question_a += f"Title: {title}\n"
        question += f"Title: {title}\n"
    question_a += context
    question += context
    # question += f'\nQuestion: {sample["question"]}'
    question_a += f'\n{sample["question"]}'
    question += f'\n{sample["question"]}'
    return question, question_a

def make_test_prompt_v2(sample, context, title=None):
    prompt = SYSTEM_PROMPT[0]
    # question_a  = "Read the passage and judge if the asked question can be answered. Only answer yes or no.\n"
    # question = "Read the passage and give me a brief answer. \n"
    if title is not None:
        prompt += "\nTitle: " + title

    prompt += "\nBackground: " + context
    question_first = prompt + "\nQuestion: " + sample["question"] + "\n" + CHECK_QUESTION + "\nAnswer:"
    question_final = prompt + "\nQuestion: " + sample["question"] + ANSWER_DEMAND
    return sample["question"], question_first, question_final


def get_label(sample):
    is_impossible = sample["is_impossible"]
    # import ipdb; ipdb.set_trace()
    if is_impossible:
        labels = [""]
    else:
        labels = sample["answers"]
    return labels, is_impossible


def evaluate_squadv2(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, add_bos_token=False, model_max_length=4096, padding_side="right", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task)):
        os.makedirs(os.path.join(args.save_dir, "test_results_{}_".format(args.model) + args.task))

    dataset = json.load(open(args.test_data_path, "r"))["data"]

    stop_str = SEP
    keywords = [stop_str]

    total = 0
    cors, incs, f1s = [], [], []
    questions, labels = [], []
    tic = time.time()
    logger.info("Starting evaluation")
    for doc in tqdm(dataset):
        title = doc["title"]
        for para in doc["paragraphs"]:
            context = para["context"]
            for item in para["qas"]:
                # question, question_a = make_test_prompt(item, context=context, title=title)
                raw_question, question_first, question_final = make_test_prompt_v2(item, context=context, title=title)
                label, is_impossible = get_label(item)

                # if len(_labels) > 1:
                #     label = _labels
                # else:
                #     label = [""]
                questions.append(raw_question)
                labels.append(label)

                # check_question = question + CHECK_QUESTION
                # conv = default_conversation.copy()
                # conv.append_message(conv.roles[0], check_question)
                # conv.append_message(conv.roles[1], None)
                # prompt = conv.get_prompt()
                # prompt = question_a + CHECK_QUESTION
                prompt = question_first

                input_ids_check = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids_check)
                with torch.no_grad():
                    raw_results = model.generate(input_ids_check, do_sample=False, temperature=0., max_new_tokens=100, stopping_criteria=[stopping_criteria])
                raw_results = tokenizer.decode(raw_results[0].tolist(), skip_special_tokens=True)
                # import ipdb; ipdb.set_trace()
                check_result = raw_results[len(prompt) + 1: ].strip()

                # print(check_result, is_impossible)
                # print("====================================")
                # if "no" in check_result.lower() or ("yes" not in check_result.lower()) or "not" in check_result.lower():
                # import ipdb; ipdb.set_trace()
                # if "no" in check_result.lower() or "not" in check_result.lower():
                if "not in the background" in check_result.lower() or "no," in check_result.lower() or "no." in check_result.lower():
                    results = [""]
                else:
                    # conv = default_conversation.copy()
                    # conv.append_message(conv.roles[0], question + "Please provide an accurate, short answer.")
                    # conv.append_message(conv.roles[1], "The answer is:")
                    # prompt = conv.get_prompt()
                    # if prompt.endswith(stop_str):
                    #     prompt = prompt[: -len(stop_str)]
                    prompt = question_final
                    # prompt = question + "Please provide an accurate, short answer. The answer is:"

                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                    with torch.no_grad():
                        raw_results = model.generate(input_ids, do_sample=False, temperature=0., max_new_tokens=100, stopping_criteria=[stopping_criteria])
                    raw_results = [tokenizer.decode(raw_results[0].tolist(), skip_special_tokens=True)]
                    # if "\n\nAssistant:" in questions[0]:
                    # import ipdb; ipdb.set_trace()
                    # results = [i[len(prompt) + 1:].strip().replace('"', '').replace('.', '') for i in raw_results]
                    results = [i[len(prompt) + 1:].strip().replace('.', '') for i in raw_results]
                    # print(results, label)
                    # for i in range(len(results)):
                    #     if results[i].endswith("."):
                    #         results[i] = results[i][:-1]
                    # results = [i.split("Assistant:")[0].strip() for i in raw_results]
                    # else:
                    #     results = [i.split("Answer:")[1].split("Question:")[0].strip() for i in raw_results]

                print("Question: ", raw_question)
                print("Answer: ", results[0])
                if label[0] == "":
                    print("GT: NO ANSWER")
                else:
                    print("GT: ", label[0]["text"])
                print("====================================")
                # logger.info(f"Question: {question}")
                # logger.info(f"Answer: {results[0]}")
                # if label[0] == "":
                #     logger.info("GT: None")
                # else:
                #     logger.info(f'GT: {label[0]["text"]}')
                cor = [exact_match(result, label) for result, label in zip(results, labels)]
                inc = [include_answer(result, label) for result, label in zip(results, labels)]
                f1s.extend([f1_score(result, label) for result, label in zip(results, labels)])
                cors.extend(cor)
                incs.extend(inc)
                questions.clear()
                labels.clear()

                total += 1  # Update the total number of predictions
                if total % 5 == 0:
                    # Print the accuracy
                    _res = {"include": np.mean(incs), "accuracy": np.mean(cors), "F1 score": np.mean(f1s)}
                    logger.info("Include: {}, EM score: {}, F1 score: {}".format(np.mean(incs), np.mean(cors), np.mean(f1s)), end="\r")
                    del _res

    res = {"include": np.mean(incs), "accuracy": np.mean(cors), "F1 score": np.mean(f1s)}
    logger.info("Include: {}, Accuracy: {}, F1 score: {}".format(res["include"], res["F1 score"], res["F1 score"]), end="\r")

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
        default="./data/language_official_benchmarks/squadv2/squad-dev-v2.0.json",
    )
    parser.add_argument("--task", "-t", type=str, default="squadv2")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="llama-vicuna-7b")
    parser.add_argument("--model_dir", type=str, default="path2model/llama-vicuna-7b-v1.1")
    args = parser.parse_args()
    evaluate_squadv2(args)
