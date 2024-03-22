#!/usr/bin/env python3

import re
import os
import sqlite3
import sys
import torch
import argparse
import numpy as np
from omni.utils.loguru import logger
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict

from datasets import load_dataset
from llama import LLaMA

from llama_evaluation_main.llama_evaluation.utils import (
    setup_model_parallel, is_equiv, get_answer_str, is_latex_equal,
    get_dataset_info, ensure_model_info_exist, get_max_eval_count,
    insert_data, metrics_to_database, DATABASE_PATH,
    seed_everything, add_common_args, modelname_by_path,
    load_model_by_args, sft_prompt, sft_answer,
    tokenize, monkey_patch_llama, random_choice_data
)


def dump_result(
    outputs, answers, types, levels, cors, level_cors, subject_cors, correct, total,
) -> Dict[str, float]:
    # log and write to file
    metrics = {}
    subjects = ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']
    split_string = "#####################"

    for k, (output, answer, prob_type, prob_level) in enumerate(zip(outputs, answers, types, levels)):
        logger.info(f"{k} TYPE: {prob_type} | LEVEL: {prob_level} | OUTPUT: {output} | ANSWER: {answer}\n")

    for prob_type in subjects:
        for prob_level in [1, 2, 3, 4, 5]:
            if (prob_level, prob_type) in cors:
                cors_list = cors[(prob_level, prob_type)]
                logger.info("{} Level {} Accuracy = {}/{} = {:.3f}".format(prob_type, prob_level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
    logger.info(split_string)

    # also get accuracies for each problem type
    for level in sorted(level_cors):
        cors_list = level_cors[level]
        acc = np.mean(cors_list)
        logger.info("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), acc))
        metrics[f"level_{level}_acc"] = acc
    logger.info(split_string)

    for subject in subjects:
        # for subject in sorted(subject_cors):
        if subject in subject_cors:
            cors_list = subject_cors[subject]
            acc = np.mean(cors_list)
            logger.info("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), acc))
            metrics[f"{subject}_acc"] = acc
    logger.info(split_string)

    overall_acc = correct / total
    logger.info(f"Overall Accuracy = {correct}/{total} = {overall_acc:.3f}")
    metrics["overall_acc"] = overall_acc
    return metrics


def generate_train_prompt(sft: bool = False) -> str:
    from llama_evaluation.prompts.math_4shot import MATH_PROMPT, SFT_MATH_PROMPT
    if sft:
        return SFT_MATH_PROMPT
    else:
        return MATH_PROMPT


def prompt_enginering(problem: str, train_prompt: str, sft: bool = False) -> str:
    if sft:
        return train_prompt + sft_prompt(problem)
    else:
        main_prompt = f"{train_prompt}Problem:\n{problem}"
        return f"{main_prompt}\n\nSolution:\n"


def get_final_answer(output: str) -> str:
    matches = re.findall(r"The final answer is (-?\d+\.?\d*)", output)
    if matches:
        return matches[0].strip(".")
    else:
        return get_answer_str(output)


def extract_generated_answer(output: str, sft: bool = False) -> str:
    if sft:
        answer = sft_answer(output)
    else:
        # 5 is magic number and depends on the number of prompts
        try:
            answer = output.split("Solution:\n")[5]
        except Exception:
            answer = output.split("Solution:\n")[-1]
    return answer


def major_vote_evaluate(
    model: LLaMA, prompts: str, loop_length: int = 10, temperature=0.1, vote=True, sft=False
) -> str:
    answer_dict = defaultdict(lambda: 0)
    for _ in tqdm(range(loop_length)):
        output_str = model.generate(
            prompts, temperature=temperature, max_gen_len=1024, apply_prompts_as_token=sft,
        )
        output_full = []
        for x in output_str:
            y = extract_generated_answer(x, sft=sft)
            # logger.info(f"Extract output:\n{x}")
            output_full.append(y)

        final_answers = [get_final_answer(x) for x in output_full]
        if not vote:
            return output_full, final_answers

        for val in final_answers:
            answer_dict[val] += 1

    # sort dict by value
    answer_dict = dict(sorted(answer_dict.items(), key=lambda x: x[1], reverse=True))
    for k, v in answer_dict.items():
        logger.info(f"Value {k}: Count: {v}")

    # major vote, return key with the max value in counter_dict
    vote_answer = max(answer_dict, key=answer_dict.get)
    if vote_answer is None:
        answer_dict.pop(None)
        vote_answer = max(answer_dict, key=answer_dict.get)
    return output_full, vote_answer


def solution2answer(solution: str, math_mode="eval_peeking") -> str:
    answer = solution
    if math_mode == "eval_peeking":
        answer = get_answer_str(solution)
    else:
        raise ValueError(f"Invalid math_mode: {math_mode}")
    return answer


def is_equal(str1, str2):
    first_equal = is_equiv(str1, str2)
    if first_equal:
        return True
    return is_latex_equal(str1, str2)


def math2db(model_id, data_list: List[Dict], metrics=None, check=False):
    dataset_id = get_dataset_info("math", field="id")
    assert dataset_id, "dataset 'math' not found"

    # get eval count number from eval db
    eval_count = get_max_eval_count(model_id, dataset_name="math") + 1
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, "evaluate_info.sqlite"))
    logger.info(f"eval count: {eval_count}")
    keys2check = ["model_id", "dataset_id", "eval_count", "sample_id", "output"]

    if metrics is not None:
        metrics_to_database(metrics, model_id, dataset_id, eval_count, percentage=True)

    # batch write
    for data in data_list:
        result = data["sample_id"], data["pred_answer"], data["short_answer"], data["equiv"]  # noqa
        data2db = (model_id, dataset_id, eval_count, *result)  # noqa
        insert_data(conn, "math_eval", data2db, check=check, keys2check=keys2check)
        conn.commit()
    conn.close()


@torch.no_grad()
def evaluate_math(
    model: LLaMA, dataset, train_prompt,
    math_mode: str = "eval_peeking", sft: bool = False,
    batch_size: int = 32, temperature: float = 0.1, local_rank: int = 0,
):
    """Evaluate the LLM on the math dataset.

    Args:
        math_mode (str, optional): mode to evaluate math dataset. Defaults to "eval_peeking".
        sft (bool, optional): Whether to use sft prompt. Defaults to False.
        batch_size (int, optional): Batch size. Defaults to 32.
        temperature (float, optional): Temperature for sampling. Defaults to 0.1.
    """
    outputs, answers, types, levels, data_list = [], [], [], [], []
    cors, subject_cors, level_cors = defaultdict(list), defaultdict(list), defaultdict(list)
    correct, total, sample_id = 0, 0, 1

    chosen_idx = random_choice_data(dataset)

    dataset_len = len(chosen_idx)
    logger.info(f"Dataset length: {dataset_len}")

    for idx, data_idx in enumerate(chosen_idx, 1):
        logger.info(f"idx: {idx} data idx: {data_idx}")
        data = dataset[data_idx]
        problem, solution = data["problem"], data["solution"]
        prob_level, prob_type = data["level"], data["type"]
        answer = solution2answer(solution, math_mode)
        logger.info(f"Prompt String:\n{problem}")
        problem = prompt_enginering(problem, train_prompt, sft=sft)
        if sft:
            problem = tokenize(problem, model.tokenizer)

        # prepare data
        if idx % batch_size == 1:
            batch_prompt, batch_answer, batch_type, batch_level = [problem], [answer], [prob_type], [prob_level]  # noqa
        else:
            batch_prompt.append(problem)
            batch_answer.append(answer)
            batch_type.append(prob_type)
            batch_level.append(prob_level)

        if idx % batch_size != 0 and idx != dataset_len:
            continue
        else:
            if len(batch_prompt) != batch_size:
                assert idx == dataset_len
            vote_output, vote_pred = major_vote_evaluate(model, batch_prompt, 1, vote=False, temperature=temperature, sft=sft)  # noqa
            outputs.extend(vote_output)
            answers.extend(vote_pred)
            types.extend(prob_type)
            levels.extend(prob_level)

            for out_str, pred_answer, answer in zip(vote_output, vote_pred, batch_answer):
                logger.info(f"Sample id: {sample_id}")
                logger.info(f"Model output:\n{out_str}")
                logger.warning(f"Correct answer: {answer}\nExtract output: {pred_answer}")

                equiv = is_equal(pred_answer, answer)
                if equiv:
                    correct += 1

                short_answer = None
                if pred_answer is not None:
                    short_answer = get_answer_str(pred_answer)
                if short_answer is None:
                    short_answer = "None"
                cors[(prob_level, prob_type)].append(equiv)
                level_cors[prob_level].append(equiv)
                subject_cors[prob_type].append(equiv)

                data_list.append({
                    "sample_id": sample_id,
                    "pred_answer": pred_answer,
                    "short_answer": short_answer,
                    "equiv": str(equiv),
                })
                sample_id += 1

            total += len(batch_prompt)
            logger.warning(f"current accuracy: {correct / total :.2%}\n")

    # log and write to file and database
    metrics = None
    if local_rank == 0:
        metrics = dump_result(
            outputs, answers, types, levels, cors,
            level_cors, subject_cors, correct, total,
        )
    return data_list, metrics


@logger.catch(reraise=True)
def main(args):
    if args.model is None:
        args.model = modelname_by_path(args.ckpt_dir)
    if args.logfile is None:
        temperature_str = f"{args.temperature:.2f}".replace(".", "_")
        logfile_dir = os.environ.get("EVAL_LOGFILE_PATH", "data/logfile")
        suffix = f"{args.model}_{temperature_str}.log"
        args.logfile = os.path.join(logfile_dir, "math", suffix)

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        logger.remove()
        sys.stdout = open(os.devnull, 'w')
    else:
        # save logger info to file
        logger.add(args.logfile, enqueue=True, backtrace=True, diagnose=True)

    logger.info(args)
    train_prompt = generate_train_prompt(sft=args.sft)

    if args.task == "math":
        dataset = load_dataset("competition_math", split="test")
    else:
        raise ValueError(f"task {args.task} not supported")

    model = load_model_by_args(args, local_rank, world_size)
    data_list, metrics = evaluate_math(
        model, dataset, train_prompt,
        math_mode=args.math_mode, sft=args.sft, batch_size=args.batch_size,
        temperature=args.temperature, local_rank=local_rank,
    )
    if local_rank == 0 and not args.no_db:
        model_id = ensure_model_info_exist(
            model_name=args.model, topk=args.topk, topp=args.topp, temperature=args.temperature
        )
        math2db(model_id, data_list, metrics=metrics)


if __name__ == "__main__":
    seed_everything(42)
    monkey_patch_llama()
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="infer batchsize")
    parser.add_argument("--num-samples", type=int, default=256, help="number of samples")
    parser.add_argument("--task", type=str, default="math", help="task name")
    parser.add_argument('--math-mode', default='eval_peeking', type=str)
    args = parser.parse_args()
    main(args)
