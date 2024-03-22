import os
import sys
import json
import time
import torch
import sqlite3
import argparse
from tqdm import tqdm
from typing import Dict
from omni.utils.loguru import logger
from types import MethodType
from functools import partial

from llama_evaluation_main.llama_evaluation.data import DATASETS
from llama_evaluation_main.llama_evaluation.evaluator import EVALUATORS
from llama_evaluation_main.llama_evaluation.utils import (
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
from omni.eval.language_eval.modeling_dreamllm import setup_model_parallel, load

from omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from omni.utils.loguru import logger
from omni.utils.profiler import FunctionProfiler
from transformers import LlamaTokenizer
from omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM as AutoModelForCausalLM

def new_tmp_file(args, use_db):
    filename = os.path.join(args.save_dir, args.model + ".tmp.jsonl")
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
    filename = os.path.join(args.save_dir, args.model + ".tmp.jsonl")
    os.remove(filename)


def test_prompt_hook(self, sample):
    sample = self.test_origin_prompt(sample)
    if isinstance(sample, str):
        return sft_prompt(sample)
    else:
        return [sft_prompt(sample[0])] + list(sample[1:])


def online_eval(task, evaluator, dataset, batch_size, addr, fs) -> Dict[str, float]:
    st = time.time()
    scores = evaluator(dataset, batch_size, addr, fs)
    scores_log = ", ".join("{} {:.2%}".format(m, c) for m, c in scores.items())
    et = time.time()
    logger.info(f"Average {scores_log}, total time {et-st:2f} - {task}")
    return scores


@torch.no_grad()
def hf_eval(task, evaluator, model, tokenizer, dataset, batch_size, fs):
    st = time.time()
    scores = evaluator(dataset, tokenizer, model, fs)
    scores_log = ", ".join("{} {:.2%}".format(m, c) for m, c in scores.items())
    et = time.time()
    logger.info(f"Average {scores_log}, total time {et-st:2f} - {task}")
    return scores

def load_model(model_name_or_path):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=True,
        padding_side="left",
    )

    with torch.device("cuda"):
        config = DreamLLMConfig.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        with FunctionProfiler("AutoModelForCausalLM.from_pretrained"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                tokenizer=tokenizer,
                config=config,
                local_files_only=True,
            )
    return tokenizer, model

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
def main(args):
    if "_sft_" in args.ckpt_dir:
        args.sft = True
    if args.logfile is None:
        args.logfile = os.path.join(args.save_dir, args.model + ".log")
    if args.online:
        args.mode = "online"

    # setup pytorch parallel
    if args.mode == "local":
        local_rank, world_size = setup_model_parallel()
        # generator = load_model_by_args(args, local_rank, world_size)
        generator = load(args.ckpt_dir, world_size)
    elif args.mode == "hf":
        tokenizer, model = load_model(args.ckpt_dir)
        local_rank = 0
    elif args.mode == "online":
        args.topp = args.topk = None
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
        print("*************: ", mode_pfx, sft_pfx, task)
        evaluator = EVALUATORS[mode_pfx + sft_pfx + task.split("/")[0]]
        fs = new_tmp_file(args, use_db)

        if args.mode == "local":
            metrics = local_eval(task, evaluator, generator, dataset, args.batch_size, fs)
        elif args.mode == "online":
            metrics = online_eval(task, evaluator, dataset, args.batch_size, args.addr, fs)
        elif args.mode == "hf":
            metrics = hf_eval(task, evaluator, model, tokenizer, dataset, args.batch_size, fs)
        else:
            raise NotImplementedError
        all_metrics[task] = metrics

        if use_db:
            write2db(args, task, fs, metrics)

    # bbh_metrics = []
    # for k, v in all_metrics.items():
    #    if "bbh" in k:
    #        bbh_metrics.append(v["accuracy"])
    # if bbh_metrics:
    #    logger.info(f"BBH average accuracy: {sum(bbh_metrics) / len(bbh_metrics):.2%}")
    metrics_to_write = metrics_post_process(args.tasks.split(","), all_metrics)
    if metrics_to_write and use_db:
        for task, metrics in metrics_to_write.items():
            write_metrics_by_args(args, metrics, task, percentage=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=None, help="Model name to database.")
    parser.add_argument("--model_size", type=str, default="7b", help="Model size to load parameters.")
    parser.add_argument(
        "--load_orin", action="store_true",
        help="load origin torch mode checkpoints, default load safetensors mode"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="path2model",
        help="Path to the checkpoint directory."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="path2model",
        help="Path to the tokenizer."
    )
    parser.add_argument(
        "--addr", type=str, default=None,
        help="Address to access TGI"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=2048,
        help="Max sequence length of LLM. Default: 2048",
    )
    parser.add_argument(
        "--topk", type=int, default=None,
        help="topk value, should be positive, defalut: None"
    )
    parser.add_argument(
        "--topp", type=float, default=None,
        help="topp value, range [0, 1], defalut: None"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="model temperature value, defalut: 0.1"
    )
    parser.add_argument(
        "--sft", action="store_true",
        help="benchmark sft model, defalut not sft"
    )
    parser.add_argument(
        "--no_db", action="store_true",
        help="not using database to store evaluate result, defalut writing to database"
    )
    parser.add_argument(
        "--logfile", type=str, default=None,
        help="log file of inference result. Default is None, which means generate by script.",
    )
    parser.add_argument("--ntrain", "-k", type=int, default=0, help="Number of samples for few-shot, default: 0 (zero-shot)")

    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        default="clue_c3,clue_wsc,clue_cmrc,boolq,piqa,siqa,hellaswag,winogrande,arc_e,arc_c,obqa,race_m,race_h,naturalqa,triviaqa,drop_gen,sciq,xtreme",
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
    parser.add_argument("--save_dir", type=str, default="./results/multi_choice", help="Path to save log files")
    args = parser.parse_args()
    print(args)
    print(args.model)
    main(args)
