#!/usr/bin/env python3

import argparse
import json
import os
import sqlite3
import sys
from typing import Dict

import torch
from datasets import load_dataset
from omni.utils.loguru import logger
from tqdm import tqdm

from human_eval.data import HUMAN_EVAL, stream_jsonl, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from llama import LLaMA
from llama_evaluation_main.llama_evaluation.utils import (
    DATABASE_PATH, add_common_args,
    ensure_model_info_exist, extract_function,
    get_dataset_info, get_max_eval_count,
    insert_data, load_model_by_args,
    metrics_to_database, modelname_by_path,
    setup_model_parallel, sft_prompt
)


def humaneval_entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
    return results


def rewrite_jsonl(filename, save_name=None):
    if save_name is None:
        save_name = filename
    with open(filename, "r") as f:
        contents = [json.loads(x) for x in f.readlines()]
    with open(save_name, "w") as f:
        for result in contents:
            code = result["completion"]
            result["completion"] = extract_function(code)
            f.write(json.dumps(result) + "\n")
    print(f"write to {save_name}")


def code2db(model_id: int, filename: str, metrics: Dict[str, float] = None):
    dataset_id = get_dataset_info(field="id")
    logger.info("dataset_id:", dataset_id)

    conn = sqlite3.connect(os.path.join(DATABASE_PATH, "evaluate_info.sqlite"))
    eval_count = get_max_eval_count(model_id) + 1
    logger.info(f"eval count: {eval_count}")

    if metrics is not None:
        metrics_to_database(metrics, model_id, dataset_id, eval_count)

    result_file = os.path.basename(filename).split(".")[0] + ".jsonl_results.jsonl"
    for data in stream_jsonl(result_file):
        sample_id = int(data["task_id"].split("/")[-1])
        raw, code, passed = data["raw"], data["completion"], data["passed"]
        data2db = (model_id, dataset_id, eval_count, sample_id, raw, str(passed), code)
        insert_data(conn, "codegen_eval", data2db, check=False)


@torch.no_grad()
def evaluate_human_eval(
    model: LLaMA, dataset, num_samples_per_task: int = 1,
    sft: bool = False, temperature: float = 0.1, batch_size: int = 32,
    filename: str = "samples.jsonl", local_rank: int = 0, use_db: bool = True,
):
    """
    TODO: num_samples_per_task is not used here.

    Args:
        model_id (Optional[str]): model id in database. Only used when use_db is True.
        sft (bool): whether use supervised finetune prompt or not. Default: False.
        temperature (float): temperature for generation.
        batch_size (int): batch size for generation.
        filename (str): filename to save generated samples. Should be jsonl format.
        local_rank (int): local rank for distributed evaluation. Default: 0.
        use_db (bool): whether to use database to store evaluation results. Default: True.
    """
    code_samples = []
    dataset_len = len(dataset)

    for idx, data in enumerate(dataset, 1):
        prompt, task_id = data["prompt"], data["task_id"]
        if sft:
            prompt = "Complete the following python code. Answer in markdown format.\n```python\n" + prompt + "\n```\n"
            prompt = sft_prompt(prompt)
        if idx % batch_size == 1:
            prompt_list, task_list = [prompt], [task_id]
        else:
            prompt_list.append(prompt)
            task_list.append(task_id)

        if idx % batch_size != 0 and idx != dataset_len:
            continue
        else:
            logger.info("Generating code...")
            if len(prompt_list) != batch_size:
                assert idx == dataset_len
            # prompts = [prompt for _ in range(batch_size)]
            contents = model.generate(prompt_list, max_gen_len=1024, temperature=temperature)
            new_code = [extract_function(x, markdown=sft) for x in contents]
            for task_id, prompt, raw, code in zip(task_list, prompt_list, contents, new_code):
                logger.info(f"task_id: {task_id} prompt:\n{prompt}\nraw:\n{raw}\ncode:\n{code}")

            code_samples.extend([dict(task_id=x, raw=y, completion=z) for x, y, z in zip(task_list, contents, new_code)])  # noqa
            torch.distributed.barrier()

    if local_rank == 0:
        write_jsonl(filename, code_samples)
        metrics = humaneval_entry_point(filename)
        logger.info(metrics)
        return metrics


@torch.no_grad()
def evaluate_mbpp(model: LLaMA, dataset, num_samples_per_task=200, batch_size=10, filename="mbpp_sample.jsonl"):
    template = "You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n"  # noqa

    code_samples = []
    for idx, data in enumerate(dataset):
        text, test_list, task_id, gt_code = data["text"], data["test_list"], data["task_id"], data["code"]
        prompt = template.format(prompt=text, tests="\n".join(test_list))
        logger.info(f"idx: {idx}\ntask_id: {task_id}\nprompt: {prompt}")
        for _ in tqdm(range(num_samples_per_task // batch_size)):
            prompts = [prompt for _ in range(batch_size)]
            contents = model.generate(prompts, max_gen_len=1024)
            new_content = [extract_function(x) for x in contents]
            logger.info(f"contents:\n{new_content[0]}")
            for prev_code, code in zip(contents, new_content):
                if prev_code == code:
                    logger.warning(f"no change:\n{prev_code}")

            code_samples.extend([dict(task_id=task_id, completion=x) for x in new_content])
            torch.distributed.barrier()

    write_jsonl(filename, code_samples)
    metrics = humaneval_entry_point(filename)
    return metrics


@logger.catch(reraise=True)
def main(args):
    if args.model is None:
        args.model = modelname_by_path(args.ckpt_dir)
    if args.logfile is None:
        temperature_str = f"{args.temperature:.2f}".replace(".", "_")
        logfile_dir = os.environ.get("EVAL_LOGFILE_PATH", "data/logfile")
        suffix = f"{args.model}_{temperature_str}.log"
        args.logfile = os.path.join(logfile_dir, "codegen", suffix)

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        logger.remove()
        sys.stdout = open(os.devnull, 'w')
    else:
        # save logger info to file
        logger.add(args.logfile, enqueue=True, backtrace=True, diagnose=True)

    logger.info(args)

    eval_func = evaluate_human_eval

    if args.task == "human_eval":
        dataset = load_dataset("openai_humaneval", split="test")
    elif args.task == "mbpp":
        raise NotImplementedError

    model = load_model_by_args(args, local_rank, world_size)

    if args.temperature is None:
        args.temperature = 0.1 if args.k == 1 else 0.8  # according to LLaMA paper

    metrics = eval_func(
        model, dataset, args.num_samples,
        sft=args.sft, temperature=args.temperature,
        batch_size=args.batch_size, filename=args.jsonl_file,
    )
    if local_rank == 0 and not args.no_db:
        logger.info("Saving results to database...")
        model_id = ensure_model_info_exist(
            model_name=args.model, topk=args.topk, topp=args.topp, temperature=args.temperature
        )
        code2db(model_id, args.jsonl_file, metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("-n", "--num-samples", type=int, default=200, help="num samples per task")
    parser.add_argument("-b", "--batch_size", type=int, default=5, help="infer batch size")
    parser.add_argument("-k", type=int, default=100, help="num samples per task")
    parser.add_argument("--task", type=str, default="human_eval", help="task name")
    parser.add_argument("--jsonl_file", type=str, default="samples.jsonl")
    args = parser.parse_args()
    main(args)
