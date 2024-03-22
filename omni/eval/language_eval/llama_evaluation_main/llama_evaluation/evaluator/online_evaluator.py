import re
import tqdm
import numpy as np
from omni.utils.loguru import logger

from concurrent.futures import ThreadPoolExecutor
from .evaluator_utils import (
    online_forward,
    get_logprobs_by_labels,
    exact_match,
    include_answer,
    f1_score,
    post_process,
    write_jsonl,
    per_sentence_exact_match,
)


__all__ = [
    "online_boolq_options_logprob_evaluator",
    "online_generation_for_choice_evaluator",
    "online_options_logprob_evaluator",
    "online_wsc_options_logprob_evaluator",
    "online_norm_by_answer_evaluator",
    "online_predict_last_evaluator",
    "online_short_generation_evaluator",
    "online_short_generation_cn_evaluator",
    "online_poem_generation_evaluator",
    "online_multitask_generation_evaluator",
    "online_longcontext_generation_cn_evaluator",
    "online_easymmlu_options_logprob_evaluator",
    "online_hypereval_evaluator",
]


def online_boolq_options_logprob_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, _) = args
            options = dataset.labels
            res_opts = []
            for opt in options:
                opt = opt.strip()
                res = online_forward(param_args, " ".join([prompt, opt]), ip_port, 1)
                res = get_logprobs_by_labels(res[1], opt)
                res_opts.append(res)
            return idx, np.array(res_opts)

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        label = dataset[idx][-1]
        cors += label == np.argmax(result)
        write_jsonl(fs, [idx + 1, int(np.argmax(result)), str(result.tolist()), str(label == np.argmax(result))])
    return {"accuracy": cors / len(dataset)}


def online_easymmlu_options_logprob_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, _) = args
            options = dataset.labels
            res_opts = []
            for opt in options:
                opt = opt.strip()
                res = online_forward(param_args, " ".join([prompt, opt]), ip_port, 1)
                res_opts.append(res[1][-1]["logprob"])
            return idx, np.array(res_opts)

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        label = dataset[idx][-1]
        cors += label == np.argmax(result)
        write_jsonl(fs, [idx + 1, int(np.argmax(result)), str(result.tolist()), str(label == np.argmax(result))])
    return {"accuracy": cors / len(dataset)}


def online_options_logprob_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, options, _) = args
            res_opts = []
            for opt in options:
                opt = opt.strip()
                res = online_forward(param_args, " ".join([prompt, opt]), ip_port, 1)
                res = get_logprobs_by_labels(res[1], opt) / len(opt)
                res_opts.append(res)
            return idx, np.array(res_opts)

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        label = dataset[idx][-1]
        cors += label == np.argmax(result)
        write_jsonl(fs, [idx + 1, int(np.argmax(result)), str(result.tolist()), str(label == np.argmax(result))])
    return {"accuracy": cors / len(dataset)}


def online_wsc_options_logprob_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, options, _) = args
            res_opts = []
            for opt in options:
                opt = opt.strip()
                res = online_forward(param_args, " ".join([prompt, opt]), ip_port, 1)
                res = get_logprobs_by_labels(res[1], opt) / len(opt)
                res_opts.append(res)
            return idx, np.array(res_opts)

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        label = dataset[idx][-1]
        cors += label == (np.argmax(result) % 2)
        write_jsonl(fs, [idx + 1, int(np.argmax(result) % 2), str(result.tolist()), str(label == np.argmax(result) % 2)])
    return {"accuracy": cors / len(dataset)}


def online_norm_by_answer_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, options, label) = args
            res_opts = []
            for opt in options:
                opt = opt.strip()
                res1 = online_forward(param_args, " ".join([prompt, opt]), ip_port, 1)
                res1 = get_logprobs_by_labels(res1[1], opt)
                res2 = online_forward(param_args, " ".join(["Answer:", opt]), ip_port, 1)
                res2 = get_logprobs_by_labels(res2[1], opt)
                res_opts.append(res1 - res2)
            return idx, np.array(res_opts)

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        label = dataset[idx][-1]
        cors += label == np.argmax(result)
        write_jsonl(fs, [idx + 1, int(np.argmax(result)), str(result.tolist()), str(label == np.argmax(result))])
    return {"accuracy": cors / len(dataset)}


def online_predict_last_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, options, _) = args
            res_opts = []
            post_ctx = options.pop()
            for opt in options:
                pmt = prompt + " " + opt
                res = online_forward(param_args, " ".join([pmt, post_ctx]), ip_port, 1)
                res = get_logprobs_by_labels(res[1], post_ctx)
                res_opts.append(res)
            return idx, np.array(res_opts)

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        label = dataset[idx][-1]
        cors += label == np.argmax(result)
        write_jsonl(fs, [idx + 1, int(np.argmax(result)), str(result.tolist()), str(label == np.argmax(result))])
    return {"accuracy": cors / len(dataset)}


def online_short_generation_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, label) = args
            res = online_forward(param_args, prompt, ip_port, 100)[0]
            res = res.split("Question:")[0].strip()
            return idx, res

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = f1s = incs = 0
    for idx, result in results:
        labels = dataset[idx][-1]
        cors += exact_match(result, labels)
        incs += include_answer(result, labels)
        f1s += f1_score(result, labels)
        write_jsonl(fs, [idx + 1, result, str(exact_match(result, labels)), str(include_answer(result, labels)), result])
    return {
        "include": incs / len(dataset),
        "accuracy": cors / len(dataset),
        "f1_score": f1s / len(dataset),
    }


def online_short_generation_cn_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, _) = args
            res = online_forward(param_args, prompt, ip_port, 100)[0]
            res = res.split("问题：")[0].strip()
            return idx, res

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = f1s = incs = 0
    for idx, result in results:
        labels = dataset[idx][-1]
        cors += exact_match(result, labels, en=False)
        incs += include_answer(result, labels, en=False)
        f1s += f1_score(result, labels, en=False)
        write_jsonl(fs, [idx + 1, result, str(exact_match(result, labels, en=False)), str(include_answer(result, labels, en=False)), result])
    return {
        "include": incs / len(dataset),
        "accuracy": cors / len(dataset),
        "f1_score": f1s / len(dataset),
    }


def online_generation_for_choice_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, _) = args
            res = online_forward(param_args, prompt, ip_port, 5)[0]
            return idx, res

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    choices = dataset.labels
    for idx, result in results:
        label = choices[dataset[idx][-1]]
        res = post_process(result, choices)
        cors += res == label
        write_jsonl(fs, [idx + 1, res, result, str(res == label)])
    return {
        "accuracy": cors / len(results),
    }


def online_poem_generation_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, _) = args
            res = online_forward(param_args, prompt, ip_port, 30)[0]
            res = res.strip()
            return idx, res

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        labels = dataset[idx][-1]
        cors += per_sentence_exact_match(result, labels)
        logger.info(dataset[idx][0])
        logger.info(dataset[idx][-1])
        logger.info(result)
    return {
        "per_centence_acc": cors / len(dataset),
    }


def online_multitask_generation_evaluator(param_args, dataset, workers, ip_port, fs):
    import json

    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, _) = args
            res = online_forward(param_args, prompt, ip_port, dataset.gen_length)[0]
            return idx, res

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        label = dataset[idx][-1]
        cor = dataset.metrics(result, label)
        cors += int(cor)
        logger.info(f'Q: "{dataset[idx][0]}", A: "{result}", F: "{dataset.post_process(result)}", A: "{label}"')

        write_jsonl(fs, [idx + 1, dataset.post_process(result), result, str(cor)])

    return {
        "accuracy": cors / len(dataset),
    }


def online_longcontext_generation_cn_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, label) = args
            res = online_forward(param_args, prompt, ip_port, 100)[0]
            # print(res)
            question = dataset[idx][0].split("问题：")[-1]
            if "提取出几个小标题" in question:
                print(res, label)
            return idx, res

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = f1s = incs = 0
    inc_count = 0
    for idx, result in results:
        labels = dataset[idx][-1]
        question = dataset[idx][0].split("问题：")[-1]
        if "概括阅读材料" not in question and "提取出几个小标题" not in question:
            cors += exact_match(result, labels, en=False)
            incs += include_answer(result, labels, en=False)
            inc_count += 1
        f1s += f1_score(result, labels, en=False)
    return {
        "include": incs / inc_count,
        "accuracy": cors / inc_count,
        "f1_score": f1s / len(dataset),
    }


def online_hypereval_evaluator(param_args, dataset, workers, ip_port, fs):
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def eval_sample(args):
            idx, (prompt, _) = args
            res = online_forward(param_args, prompt, ip_port, 128)[0]
            return idx, res

        results = list(tqdm.tqdm(executor.map(eval_sample, enumerate(dataset)), total=len(dataset)))

    cors = 0
    for idx, result in results:
        eval_func = dataset[idx][-1]
        try:
            cor = float(eval_func(result))
        except:
            cor = 0
        cors += cor
        logger.info(f"question: [{dataset[idx][0]}]\nevalfunc: [{dataset[idx][-1]}]\nanswer: [{result}]\nscore: {cor}")

        # write_jsonl(fs, [idx+1, dataset.post_process(result), result, str(cor)])

    return {
        "accuracy": cors / len(dataset),
    }
