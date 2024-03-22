import os
import sys
import json
import time
import torch
import sqlite3
import argparse

from typing import Dict
from omni.utils.loguru import logger
from functools import partial

from llama_evaluation_main.llama_evaluation.data import DATASETS
from llama_evaluation_main.llama_evaluation.evaluator import EVALUATORS
from llama_evaluation_main.llama_evaluation.utils import (
    setup_model_parallel,
    add_common_args,
    modelname_by_path,
    dataset_mapping,
    load_model_by_args,
    sft_prompt,
    get_dataset_info,
    ensure_model_info_exist,
    get_max_eval_count,
    metrics_to_database,
    insert_data,
    DATABASE_PATH,
    metrics_post_process,
    write_metrics_by_args,
)


def new_tmp_file(args, use_db):
    filename = os.path.join(args.save_dir, "multi_choice", args.model + ".tmp.jsonl")
    if use_db:
        return open(filename, "w+")
    else:
        return None


def write2db(args, task, fs, metrics):
    model_id = ensure_model_info_exist(model_name=args.model, topk=args.topk, topp=args.topp, temperature=args.temperature)
    dataset_id = get_dataset_info(task, field="id")
    task_type = get_dataset_info(task, field="task_type") + "_eval"
    eval_count = get_max_eval_count(model_id, task)
    eval_count += 0 if args.check_repeat else 1
    data2db = [model_id, dataset_id, eval_count]
    keys2check = ["model_id", "dataset_id", "eval_count", "sample_id"]

    metrics_to_database(metrics, *data2db, percentage=True)
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, "evaluate_info.sqlite"))
    fs.seek(0)
    for i in fs.readlines():
        data = data2db + json.loads(i)
        insert_data(conn, task_type, data, check=args.check_repeat, keys2check=keys2check)
    conn.commit()
    conn.close()
    fs.close()
    filename = os.path.join(args.save_dir, "multi_choice", args.model + ".tmp.jsonl")
    os.remove(filename)


def online_eval(param_args, task, evaluator, dataset, batch_size, addr, fs) -> Dict[str, float]:
    st = time.time()
    scores = evaluator(param_args, dataset, batch_size, addr, fs)
    scores_log = ", ".join("{} {:.2%}".format(m, c) for m, c in scores.items())
    et = time.time()
    logger.info(f"Average {scores_log}, total time {et-st:2f} - {task}")
    return scores


@torch.no_grad()
def hf_eval(task, evaluator, model, tokenizer, dataset, fs):
    st = time.time()
    scores = evaluator(dataset, tokenizer, model, fs)
    scores_log = ", ".join("{} {:.2%}".format(m, c) for m, c in scores.items())
    et = time.time()
    logger.info(f"Average {scores_log}, total time {et-st:2f} - {task}")
    return scores


@torch.no_grad()
def local_eval(task, evaluator, generator, dataset, batch_size, fs):
    st = time.time()
    batch_size = 1 if dataset.forward_once else batch_size
    scores = evaluator(dataset, generator, batch_size, fs)
    scores_log = ", ".join("{} {:.2%}".format(m, c) for m, c in scores.items())
    et = time.time()
    logger.info(f"Average {scores_log}, total time {et-st:2f} - {task}")
    return scores


@logger.catch(reraise=True)
def multich_online_eval(args):
    # setup default arguments
    args.save_dir = os.environ.get("EVAL_LOGFILE_PATH", "data/logfile")
    if args.model is None:
        args.model = modelname_by_path(args.ckpt_dir)
    if "_sft_" in args.model:
        args.sft = True
    if args.logfile is None:
        args.logfile = os.path.join(args.save_dir, "multi_choice", args.model + ".log")
    if args.online:
        args.mode = "online"

    # setup pytorch parallel
    if args.mode == "local":
        local_rank, world_size = setup_model_parallel()
        generator = load_model_by_args(args, local_rank, world_size)
    elif args.mode == "hf":
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.ckpt_dir,
            trust_remote_code=True,
            use_fast=False,
        )
        local_rank = 0
    elif args.mode == "online":
        local_rank = 0
    else:
        raise NotImplementedError

    # setup logger
    if local_rank > 0:
        logger.remove()
        sys.stdout = open(os.devnull, "w")
    else:
        # save logger info to file
        logger.add(args.logfile, enqueue=True, backtrace=True, diagnose=True)

    logger.info(args)

    use_db = (not args.no_db) and (local_rank == 0)
    mode_pfx = args.mode + "_"
    sft_pfx = "sft_" if args.sft else ""

    if args.falcon_small:
        tasks = "hellaswag,winogrande,piqa,arc_e,boolq,sciq,clue_c3,clue_cmrc,xtreme,race_m,triviaqa,drop_gen".split(",")
    else:
        tasks = args.tasks.split(",")

    tasks = dataset_mapping(tasks)

    all_metrics = {}
    for task in tasks:
        # task = "bbh/subset", use "bbh" to get dataset and evaluator and subset to choose pattern and metrics
        dataset_class = DATASETS[sft_pfx + task.split("/")[0]]
        if "/" in task:
            dataset_class = partial(dataset_class, subset=task.split("/")[1])
        dataset = dataset_class(ntrain=args.ntrain, lite=args.lite, sft=args.sft)
        evaluator = EVALUATORS[mode_pfx + sft_pfx + task.split("/")[0]]
        fs = new_tmp_file(args, use_db)

        if args.mode == "local":
            metrics = local_eval(task, evaluator, generator, dataset, args.batch_size, fs)
        elif args.mode == "online":
            param_args = {
                "model": args.ckpt_dir,
                "temperature": args.temperature,
                "top_p": args.topp,
                "top_k": args.topk,
                "do_sample": args.topp is not None or args.topp is not None,
            }
            metrics = online_eval(param_args, task, evaluator, dataset, args.batch_size, args.addr, fs)
        elif args.mode == "hf":
            metrics = hf_eval(task, evaluator, model, tokenizer, dataset, fs)
        else:
            raise NotImplementedError
        all_metrics[task] = metrics

        if use_db:
            write2db(args, task, fs, metrics)

    metrics_to_write = metrics_post_process(args.tasks.split(","), all_metrics)
    if metrics_to_write and use_db:
        for task, metrics in metrics_to_write.items():
            write_metrics_by_args(args, metrics, task, percentage=True)


def multich_online_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("--ntrain", "-k", type=int, default=0, help="Number of samples for few-shot, default: 0 (zero-shot)")
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        default="clue_c3,clue_cmrc,boolq,piqa,siqa,hellaswag,winogrande,arc_e,arc_c,race_m,race_h,naturalqa,triviaqa,xtreme,ceval,mmlu",
        help='Please add tasks as: "boolq siqa"',
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the generation task, default: 32")
    parser.add_argument(
        "--check_repeat",
        default=False,
        action="store_true",
        help="If the sample has already been writen in the database, the sample will be skip, default not check",
    )
    parser.add_argument("--online", default=False, action="store_true", help="Use TGI online server, default: False")
    parser.add_argument(
        "--mode",
        default="local",
        choices=["local", "online", "hf"],
    )
    parser.add_argument("--lite", default=False, action="store_true", help="Select the first 500 samples for light validation, default validate all samples")
    parser.add_argument("--falcon_small", default=False, action="store_true", help="Run falcon small set, default: False")
    # parser.add_argument(
    #    "--save_dir", type=str, default="data/logfile/multi_choice",
    #    help="Path to save log files"
    # )
    return parser


if __name__ == "__main__":
    args = multich_online_parser().parse_args()
    multich_online_eval(args)
