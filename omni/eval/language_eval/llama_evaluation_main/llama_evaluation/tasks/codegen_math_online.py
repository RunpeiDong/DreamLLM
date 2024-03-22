#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
import functools
import requests
import os
import argparse
from omni.utils.loguru import logger
from tqdm import tqdm
from typing import List, Dict

from datasets import load_dataset
import time

from human_eval.data import write_jsonl

from llama_evaluation_main.llama_evaluation.utils import add_common_args, ensure_model_info_exist
from llama_evaluation_main.llama_evaluation.evaluator.evaluator_utils import online_forward
from llama_evaluation_main.llama_evaluation.tasks.codegen import extract_function, humaneval_entry_point, code2db
from llama_evaluation_main.llama_evaluation.prompts import Prompt, MATH_PROMPT, SFT_MATH_PROMPT, GSM8K_PROMPT, SFT_GSM8K_PROMPT, SFT_TEMPLATE, BASE_TEMPLATE

from llama_evaluation_main.llama_evaluation.tasks.math_eval import solution2answer, random_choice_data, get_final_answer, is_equal, math2db, dump_result
from collections import defaultdict


def join_details(pred: Dict) -> str:
    if "details" in pred:
        tokens = pred["details"]["tokens"]
        text = "".join([x["text"] for x in tokens]).replace("</s>", "").replace("</t>", "")
    else:
        text = pred["generated_text"]
    return text


def codegen_thread_worker(
    task: Dict, ip_port: str, max_gen_len: int = 512,
    temperature: float = 0.1, sft: bool = False, do_sample: bool = True,
    details: bool = True, topp: float = None, topk: int = None, seed: int = None,
) -> Dict:
    template = "Complete the following python code. Answer in markdown format.\n```python\n{}\n```\n" if sft else None
    prompt = Prompt(template=template)
    prompt_str = prompt.render(task["prompt"])

    param_args = {"temperature": temperature,
                  "top_p": topp,
                  "do_sample": topp is not None,
                  "seed": None if topp is not None else seed}
    raw_answer = online_forward(
        param_args, prompt_str, ip_port, max_gen_length=max_gen_len,
    )
    text = raw_answer[0]

    content = prompt_str + text
    new_code = extract_function(content, markdown=sft)
    jsonl_dict = {
        "task_id": task["task_id"],
        "completion": new_code,
        "raw": content,
    }
    return jsonl_dict


def math_thread_worker(
    data: Dict, ip_port: str, max_gen_len: int = 512,
    temperature: float = 0.1, sft: bool = False, do_sample: bool = True,
    details: bool = True, topp: float = 0.95, seed: int = None,
) -> Dict:
    if "problem" in data.keys():
        problem, solution = data["problem"], data["solution"]
        gt_answer = solution2answer(solution)
        if sft:
            prompt = Prompt(prefix=SFT_MATH_PROMPT, template=SFT_TEMPLATE)
        else:
            prompt = Prompt(prefix=MATH_PROMPT, template=BASE_TEMPLATE)
    else:
        problem, solution = data["question"], data["answer"]
        gt_answer = solution.split("#### ")[-1]
        if sft:
            prompt = Prompt(prefix=SFT_GSM8K_PROMPT, template=SFT_TEMPLATE)
        else:
            prompt = Prompt(prefix=GSM8K_PROMPT, template=BASE_TEMPLATE)

    problem = prompt.render(problem)

    param_args = {"temperature": temperature,
                  "top_p": topp,
                  "do_sample": topp is not None,
                  "seed": None if topp is not None else seed}
    raw_answer = online_forward(
        param_args, problem, ip_port, max_gen_length=max_gen_len,
    )
    raw_answer = raw_answer[0]
    short_answer = get_final_answer(raw_answer)
    if short_answer is None:
        short_answer = "None"

    equiv = is_equal(short_answer, gt_answer)

    jsonl_dict = {
        "problem": problem,
        "raw_answer": raw_answer,
        "short_answer": short_answer,
        "prob_type": data["type"] if "type" in data.keys() else "Algebra",
        "prob_level": data["level"] if "level" in data.keys() else "Level 0",
        "equiv": equiv,
    }
    return jsonl_dict


def evaluate_math_online(
    model, dataset, max_gen_len: int = 512,
    sft: bool = False, temperature: float = 0.1, num_workers=1,
    filename: str = "math_samples.jsonl", seed: int = None, do_sample: bool = False,
    details=True,
) -> Dict[str, float]:
    f = functools.partial(
        math_thread_worker, ip_port=model, temperature=temperature,
        sft=sft, do_sample=do_sample, seed=seed, details=details, max_gen_len=max_gen_len,
    )
    logger.info(f"math args:\nsft: {sft}\ntemperature: {temperature}\ndo_sample: {do_sample}\nseed: {seed}\nmax_gen_len: {max_gen_len}")  # noqa

    outputs, answers, types, levels = [], [], [], []
    cors, subject_cors, level_cors = defaultdict(list), defaultdict(list), defaultdict(list)

    chosen_idx = random_choice_data(dataset)
    dataset = [dataset[x] for x in chosen_idx]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        math_samples = list(tqdm(executor.map(f, dataset), total=len(dataset)))
    write_jsonl(filename, math_samples)

    outputs, answers, types, levels = [], [], [], []
    cors, subject_cors, level_cors = defaultdict(list), defaultdict(list), defaultdict(list)
    correct, total = 0, len(math_samples)
    for sample in math_samples:
        outputs.append(sample["raw_answer"])
        answers.append(sample["short_answer"])
        prob_type, prob_level, equiv = sample["prob_type"], sample["prob_level"], sample["equiv"]
        types.append(prob_type)
        levels.append(prob_level)
        cors[(prob_level, prob_type)].append(equiv)
        level_cors[prob_level].append(equiv)
        subject_cors[prob_type].append(equiv)
        if equiv:
            correct += 1

    metrics = dump_result(outputs, answers, types, levels, cors, level_cors, subject_cors, correct, total)
    return metrics


def evaluate_humaneval_online(
    model, dataset, num_samples_per_task: int = 1, max_gen_len: int = 512,
    sft: bool = False, temperature: float = 0.1, num_workers=1,
    filename: str = "humaneval_samples.jsonl", seed: int = None, do_sample: bool = False,
    details=True, topp=None, topk=None,
):
    template = "Complete the following python code. Answer in markdown format.\n```python\n{}\n```\n" if sft else None
    prompt = Prompt(template=template)
    prompt_str = prompt.render(dataset[-1]["prompt"])
    f = functools.partial(
        codegen_thread_worker, ip_port=model, temperature=temperature, max_gen_len=max_gen_len,
        sft=sft, do_sample=do_sample, seed=seed, details=details, topp=topp, topk=topk,
    )
    logger.info(f"humaneval args:\nsft: {sft}\ntemperature: {temperature}\ndo_sample: {do_sample}\nseed: {seed}")

    code_samples = []
    logger.info(f"Generating {num_samples_per_task} samples for every code sample...")
    for _ in range(num_samples_per_task):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            generated_code = list(tqdm(executor.map(f, dataset), total=len(dataset)))
            code_samples.extend(generated_code)

    for x in code_samples:
        logger.info(f"task_id: {x['task_id']}\ncompletion:\n{x['completion']}")
    write_jsonl(filename, code_samples)
    metrics = humaneval_entry_point(filename)
    logger.info(metrics)
    return metrics


def evaluate_mbpp_online(model, dataset, num_samples_per_task=200, temperature: float = None, batch_size=10, filename="mbpp_sample.jsonl"):
    # template = "Human: You are an expert Python programmer, and your task is to complete the following function based on it's text description:\n```\n{prompt}\n```\nYour code should pass these tests:\n```{tests}\n```\n\nAssistant:"  # noqa

    tasks = []
    for idx, data in enumerate(dataset):
        text, test_list, task_id, gt_code = data["prompt"], data["test"], data["task_id"], data["canonical_solution"]
        test_list = test_list.replace("\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\n", "")
        prompt = template.format(prompt=text, tests=test_list)
        logger.info(f"idx: {idx}\ntask_id: {task_id}\nprompt: {prompt}")
        for _ in range(num_samples_per_task):
            tasks.append({"task_id": task_id, "prompt": prompt})
    with ThreadPoolExecutor(1) as pool:
        code_samples = list(tqdm.tqdm(pool.map(functools.partial(thread_worker, ip_port=model), tasks), total=len(tasks)))

    write_jsonl(filename, code_samples)
    metrics = humaneval_entry_point(filename)
    return metrics


@logger.catch(reraise=True)
def online_eval(args):
    if args.model is None:
        assert args.no_db, "Please provide model name such as LLaMA_7B when using database"
    assert args.addr is not None, "Please provide model ip and port, example format: 192.168.0.0:8080"
    if args.logfile is None:
        temperature_str = f"{args.temperature:.2f}".replace(".", "_")
        logfile_dir = os.environ.get("EVAL_LOGFILE_PATH", "data/logfile")
        suffix = f"{args.model}_{temperature_str}.log"
        args.logfile = os.path.join(logfile_dir, args.task, suffix)

    logger.add(args.logfile, enqueue=True, backtrace=True, diagnose=True)

    logger.info("benchmark codegen task with online evaluation")
    logger.info(args)

    if args.task == "codegen":
        dataset = load_dataset("openai_humaneval", split="test")
        metrics = evaluate_humaneval_online(
            args.addr, dataset, num_samples_per_task=args.num_samples,
            max_gen_len=args.max_gen_length,
            sft=args.sft, temperature=args.temperature,
            filename=args.jsonl_file, num_workers=args.num_workers,
            seed=args.seed, do_sample=args.do_sample,
            topp=args.topp, topk=args.topk,
        )
        logger.info(metrics)
        if not args.no_db:
            logger.info("Saving results to database...")
            model_id = ensure_model_info_exist(
                model_name=args.model, topk=args.topk, topp=args.topp, temperature=args.temperature
            )
            code2db(model_id, args.jsonl_file, metrics=metrics)
    elif args.task in ["math", "gsm8k"]:
        if args.task == "math":
            dataset = load_dataset("competition_math", split="test")
        elif args.task == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="test")
        metrics = evaluate_math_online(
            args.addr, dataset, max_gen_len=args.max_gen_length,
            sft=args.sft, temperature=args.temperature,
            filename=args.jsonl_file, num_workers=args.num_workers,
            seed=args.seed, do_sample=args.do_sample,
        )
        logger.info(metrics)
        if not args.no_db:
            logger.info("Saving results to database...")
            model_id = ensure_model_info_exist(
                model_name=args.model, topk=args.topk, topp=args.topp,
                temperature=args.temperature,
            )
            math2db(model_id, args.jsonl_file, metrics=metrics)


def online_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("-n", "--num-samples", type=int, default=1, help="num samples per task")
    parser.add_argument("-w", "--num-workers", type=int, default=8, help="max workers for threadpool")
    parser.add_argument("--task", type=str, default="codegen", help="task name to benchmark")
    parser.add_argument("--seed", type=int, default=None, help="random seed for online inference")
    parser.add_argument("--do-sample", action="store_true", help="do sample or not for online inference")
    parser.add_argument("--jsonl_file", type=str, default="samples.jsonl")
    parser.add_argument("--max-gen-length", type=int, default=512)
    parser.set_defaults(topp=None, topk=None)
    return parser


if __name__ == "__main__":
    args = online_parser().parse_args()
    args.do_sample = args.topp is not None or args.topk is not None
    online_eval(args)
